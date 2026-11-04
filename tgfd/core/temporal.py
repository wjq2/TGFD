"""Temporal support for topology-guided event denoising."""
import numpy as np
from numba import njit


@njit(cache=True)
def compute_temporal_support(
    xs, ys, ts, c_ids, alive_mask, h, w, tau, pw,
    weight_mode=0,
):
    """Accumulate topology-constrained temporal support."""
    n = len(xs)
    scores = np.zeros(n, dtype=np.float32)
    first_ts = np.full(h * w, 1e18, dtype=np.float64)
    history_ts = np.full(h * w, -1e18, dtype=np.float64)
    cid_map = np.full(h * w, -1, dtype=np.int32)

    for i in range(n):
        x, y = int(xs[i]), int(ys[i])
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        p = y * w + x
        cid_map[p] = c_ids[i]
        if ts[i] < first_ts[p]:
            first_ts[p] = ts[i]

    for i in range(n):
        x, y = int(xs[i]), int(ys[i])
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        t, cid = ts[i], c_ids[i]
        if alive_mask[i]:
            support = 0.0

            for dy in range(-pw, pw + 1):
                for dx in range(-pw, pw + 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue

                    p = ny * w + nx
                    if cid_map[p] != cid:
                        continue

                    if history_ts[p] > -1e17:
                        dt = t - history_ts[p]
                    else:
                        if first_ts[p] >= 1e17:
                            continue
                        dt = abs(first_ts[p] - t)

                    if weight_mode == 0:
                        weight = tau / (tau + dt)
                    else:
                        weight = max(0.0, 1.0 - dt / tau)

                    support += weight

            scores[i] = support

        history_ts[y * w + x] = t

    return scores


def spatial_support_ebe(
    events, feat, labels, pass_mask, resolution, cn,
    *, tau_us=4000.0, radius=2, weighting="hyperbolic",
):
    """Map topology labels to events and compute temporal support."""
    if weighting not in ("hyperbolic", "linear"):
        raise ValueError("weighting must be hyperbolic or linear")
    if not np.isfinite(tau_us) or tau_us <= 0:
        raise ValueError("tau_us must be finite and positive")

    out = np.zeros(len(events), np.float32)
    h, w = resolution[1], resolution[0]
    input_valid = feat["input_valid"]
    valid_idx = np.where(input_valid)[0]
    if len(valid_idx) == 0:
        return out

    xs = feat["xs"].astype(np.int32)
    ys = feat["ys"].astype(np.int32)
    ts = events[input_valid, 0].astype(np.float64)
    c_ids = labels[cn].astype(np.int32)
    alive_mask = pass_mask[c_ids]
    weight_mode = 0 if weighting == "hyperbolic" else 1

    out[valid_idx] = compute_temporal_support(
        xs, ys, ts, c_ids, alive_mask,
        h, w, float(tau_us), int(radius), weight_mode,
    )
    return out
