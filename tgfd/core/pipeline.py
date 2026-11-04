"""Compose windowing, topology generation and temporal event scoring."""
import numpy as np
from tgfd.core.config import Config
from tgfd.io.events import validate_events
from tgfd.core.windowing import make_windows
from tgfd.core.topology import build_topology_prior
from tgfd.core.temporal import compute_history_first_future, spatial_support_ebe


def process_window(events, resolution, skip_fine=False, *, alpha=1.0,
                   min_component_size=2, tau_us=4000.0, radius=2, eps=1e-5):
    """Score one window using shared topology for full or coarse-only scoring."""
    out = np.zeros(len(events), np.float32)
    prior = build_topology_prior(events, resolution, alpha=alpha,
                                 min_component_size=min_component_size, eps=eps)
    if prior is None:
        return out
    if skip_fine:
        node_scores = np.where(prior.pass_mask[prior.labels],
                               prior.component_sizes[prior.labels].astype(np.float32), 0.0)
        out[prior.features['input_valid']] = node_scores[prior.event_node_ids]
        return out
    return spatial_support_ebe(events, prior.features, prior.labels, prior.pass_mask,
                               resolution, prior.event_node_ids, tau_us=tau_us, radius=radius)

def warmup():
    """Compile the same Numba signature used for scoring before timing a run."""
    compute_history_first_future(np.array([0, 1], np.int32), np.array([0, 0], np.int32), np.array([0.0, 1.0], np.float64), np.array([0, 0], np.int32), np.array([True, True]), 2, 2, 4000.0, 2)

def score_events(events, config=None, mode='full'):
    """Return scores in input row order, with overlap fused by maximum score.

    Each original row receives exactly one final score. Sorting follows the
    original numeric experiment scripts (NumPy quicksort); equal-time ordering
    can affect which neighbor has already been processed.
    """
    config = Config() if config is None else config
    if mode not in ('full', 'lite'):
        raise ValueError("mode must be 'full' or 'lite'")
    events = validate_events(events, config)
    if len(events) == 0:
        return np.zeros(0, np.float32)
    order = np.argsort(events[:, 0], kind='quicksort')
    ordered = events[order].copy()
    ordered[:, 0] -= ordered[0, 0]
    final = np.zeros(len(events), dtype=np.float32)
    for start, end in make_windows(ordered, config):
        if end - start < 2:
            continue
        score = process_window(ordered[start:end], (config.width, config.height), skip_fine=mode == 'lite', alpha=config.alpha, min_component_size=config.min_component_size, tau_us=config.tau_us, radius=config.radius, eps=config.eps)
        np.maximum(final[start:end], score, out=final[start:end])
    output = np.empty_like(final)
    output[order] = final
    return output
