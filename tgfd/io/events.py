"""Explicit event-column and timestamp-unit handling for NPY and AEDAT4 files."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class EventData:
    raw: np.ndarray
    events: np.ndarray
    labels: object
    columns: str
    time_unit: str


def load_events(path, columns=None, time_unit=None):
    """Load numeric events without guessing label columns or timestamp units."""
    path = Path(path)
    if path.suffix.lower() == ".aedat4":
        if columns not in (None, "t,x,y,p") or time_unit not in (None, "us"):
            raise ValueError("AEDAT4 decoding provides t,x,y,p with timestamps in microseconds")
        try:
            import aedat
        except ImportError as exc:
            raise ImportError("AEDAT4 support requires: pip install aedat") from exc
        rows = []
        for packet in aedat.Decoder(str(path)):
            if "events" in packet:
                e = packet["events"]
                rows.append(np.column_stack([e["t"], e["x"], e["y"], e["p"]]))
        raw = np.concatenate(rows).astype(np.float64) if rows else np.empty((0, 4), np.float64)
        columns, time_unit = "t,x,y,p", "us"
    elif path.suffix.lower() == ".npy":
        if columns is None or time_unit is None:
            raise ValueError("NPY input requires --columns and --time-unit; units are never guessed")
        raw = np.load(path, allow_pickle=False)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
            raise ValueError("NPY must be a real numeric (N, C) array, not a pickled object")
    else:
        raise ValueError("supported input formats: .npy and .aedat4")
    names = [c.strip() for c in columns.split(",")]
    if len(names) != raw.shape[1] or len(set(names)) != len(names):
        raise ValueError("--columns must name each input column once")
    if not {"t", "x", "y", "p"}.issubset(names) or not set(names).issubset({"t", "x", "y", "p", "label"}):
        raise ValueError("column names must be t,x,y,p and optionally label, in the actual file order")
    scales = {"s": 1e6, "ms": 1e3, "us": 1.0}
    if time_unit not in scales:
        raise ValueError("time_unit must be s, ms or us")
    events = np.asarray(raw[:, [names.index(c) for c in ("t", "x", "y", "p")]], dtype=np.float64).copy()
    # Subtract before scaling to reduce precision loss from large absolute timestamps.
    if len(events):
        events[:, 0] -= np.min(events[:, 0])
    events[:, 0] *= scales[time_unit]
    labels = raw[:, names.index("label")].copy() if "label" in names else None
    return EventData(raw, events, labels, columns, time_unit)


def signal_labels(labels, signal_value):
    """Convert two explicitly identified classes into signal=1 for ROC evaluation."""
    if labels is None:
        raise ValueError("ROC AUC requires a label column")
    values = np.unique(labels)
    if len(values) != 2 or signal_value not in values:
        raise ValueError(f"expected two classes containing signal label {signal_value}; found {values}")
    return (labels == signal_value).astype(np.int8)


def inspect_npy(path):
    """Report column ranges without assuming the data layout."""
    raw = np.load(path, allow_pickle=False, mmap_mode="r")
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("inspection requires a real numeric 2D NPY array")
    return {
        "path": str(path), "shape": list(raw.shape), "dtype": str(raw.dtype),
        "first_rows": raw[:5].tolist(),
        "columns": [
            {"index": i, "min": float(np.min(raw[:, i])) if len(raw) else None,
             "max": float(np.max(raw[:, i])) if len(raw) else None}
            for i in range(raw.shape[1])
        ],
    }


def validate_events(events, config):
    """Accept canonical [t_us, x, y, p] data and reject ambiguous coordinates."""
    events = np.asarray(events, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError('events must have shape (N, 4): [t_us, x, y, p]')
    if not np.all(np.isfinite(events)):
        raise ValueError('events must contain only finite values')
    xy = events[:, 1:3]
    if np.any(xy != np.floor(xy)):
        raise ValueError('pixel coordinates must be integers; no implicit rounding is applied')
    if np.any((xy[:, 0] < 0) | (xy[:, 0] >= config.width) | (xy[:, 1] < 0) | (xy[:, 1] >= config.height)):
        raise ValueError('pixel coordinates are outside the configured sensor resolution')
    if not np.all(np.isin(events[:, 3], (-1, 0, 1))):
        raise ValueError('polarity must use {-1, 1} or {0, 1}; p=1 is positive')
    return events
