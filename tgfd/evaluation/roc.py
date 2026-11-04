"""Signal-positive ROC evaluation following the DND21 evaluation definition.

Reference: S. Guo and T. Delbruck, Low Cost and Latency Event Camera
Background Activity Denoising, TPAMI 45(1):785-795, 2023.
DOI: 10.1109/TPAMI.2022.3152999.

This is a Python/scikit-learn adapter, not copied jAER Java code or the
paper's remote-control parameter-sweep scripts. Scores increase with signal
support. TPR=TP/(TP+FN), FPR=FP/(FP+TN).
"""
import numpy as np
from sklearn.metrics import auc, roc_curve


def compute_roc(signal_labels, scores):
    """Return all empirical ROC points, including reject-all/accept-all ends."""
    labels = np.asarray(signal_labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError('labels and scores must be aligned 1D arrays')
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError('ROC requires both classes: signal=1 and noise=0')
    if not np.all(np.isfinite(scores)):
        raise ValueError('ROC scores must be finite')
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
            'auc': float(auc(fpr, tpr))}
