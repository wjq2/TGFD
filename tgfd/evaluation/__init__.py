"""Optional score evaluation, separate from the denoising implementation."""
import numpy as np
from tgfd.evaluation.roc import compute_roc
from tgfd.evaluation.metrics import compute_mesr
from tgfd.core.filtering import make_keep_mask


def evaluate_auc(signal_labels, scores):
    """Evaluate the continuous scores for all input events."""
    return compute_roc(signal_labels, scores)['auc']


def evaluate_mesr(events, scores, resolution, *, threshold, comparison='gt',
                  block_size=30000, reference_count=None, legacy_fallback=False):
    """Report undefined filtered MESR explicitly unless legacy substitution is requested."""
    scores = np.asarray(scores)
    if scores.shape != (len(events),) or not np.all(np.isfinite(scores)):
        raise ValueError('scores must be finite and aligned with events')
    mask = make_keep_mask(scores, threshold, comparison)
    order = np.argsort(events[:, 0], kind='quicksort')
    ordered = events[order]
    raw = compute_mesr(ordered, resolution, block_size, reference_count)
    filtered = compute_mesr(ordered[mask[order]], resolution, block_size, reference_count)
    undefined = filtered is None
    if undefined and legacy_fallback:
        filtered = raw
    return dict(threshold=threshold, comparison=comparison, n_kept=int(mask.sum()),
                metric='E-MLB EventStructuralRatio', block_size=block_size,
                reference_count=(int(block_size * 2 / 3) if reference_count is None else reference_count),
                raw_mesr=raw, filtered_mesr=filtered,
                delta_mesr=filtered-raw if filtered is not None and raw is not None else None,
                filtered_mesr_undefined=undefined,
                substituted_raw_mesr=bool(undefined and legacy_fallback))
