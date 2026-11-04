"""Shared conversion from continuous event scores to a retention mask."""
import numpy as np


def make_keep_mask(scores, threshold, comparison="ge"):
    """Return an input-aligned Boolean mask using an explicit threshold rule."""
    if comparison not in ("ge", "gt"):
        raise ValueError("comparison must be ge or gt")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a finite 1D array")
    return scores >= threshold if comparison == "ge" else scores > threshold
