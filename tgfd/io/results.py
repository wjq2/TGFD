"""Output-directory handling and serialization shared by all run commands."""
import csv
import json
from pathlib import Path

import numpy as np


def prepare_output_directory(directory, inputs):
    """Create an empty destination after checking input output-name collisions."""
    directory = Path(directory)
    stems = [Path(path).stem for path in inputs]
    if len(stems) != len(set(stems)):
        raise ValueError("input filenames must have distinct stems; use separate output directories")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("output directory must be empty; choose a new one for each run")
    return directory


def save_json(path, value):
    """Write strict JSON with readable indentation."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def save_roc(directory, name, roc):
    """Save empirical ROC points, including the infinite initial threshold."""
    with (Path(directory) / f"{name}_roc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["threshold", "fpr", "tpr"])
        writer.writerows(zip(roc["thresholds"], roc["fpr"], roc["tpr"]))


def save_scores(directory, name, scores, data, config, mode):
    """Store scores and their original input-row correspondence."""
    metadata = json.dumps({
        "columns": data.columns,
        "input_time_unit": data.time_unit,
        "config": config.to_dict(),
        "mode": mode,
    })
    np.savez_compressed(
        Path(directory) / f"{name}_scores.npz",
        scores=scores,
        original_indices=np.arange(len(scores)),
        metadata=metadata,
    )


def save_filtered(directory, name, raw, mask):
    """Keep the original input layout, units and order in the filtered file."""
    directory = Path(directory)
    np.save(directory / f"{name}_keep.npy", mask)
    np.save(directory / f"{name}_denoised.npy", raw[mask])


def save_summary(directory, rows):
    """Write the same per-file run records as CSV and JSON."""
    directory = Path(directory)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_json(directory / "results.json", rows)
