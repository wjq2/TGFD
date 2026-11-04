"""Temporal support for topology-guided event denoising."""
import numpy as np
from numba import njit


@njit(cache=True)
def compute_temporal_support(
    xs, ys, ts, c_ids, alive_mask, h, w, tau, pw,
    weight_mode=0, selection_mode=1,
):
    """Accumulate support from time-ordered events in one window."""
    n = len(xs)
    scores = np.zeros(n, dtype=np.float32)
    first_ts = np.full(h * w, 1e18, dtype=np.float64)
    history_ts = np.full(h * w, -1e18, dtype=np.float64)
    cid_map = np.full(h * w, -1, dtype=np.int32)

    next_event = np.full(
        n if selection_mode == 1 else 0, -1, dtype=np.int64
    )
    cursor = np.full(
        h * w if selection_mode == 1 else 0, -1, dtype=np.int64
    )
    last_event = np.full(
        h * w if selection_mode == 1 else 0, -1, dtype=np.int64
    )

    for i in range(n):
        x, y = int(xs[i]), int(ys[i])
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        p = y * w + x
        cid_map[p] = c_ids[i]
        if ts[i] < first_ts[p]:
            first_ts[p] = ts[i]

        if selection_mode == 1:
            previous = last_event[p]
            if previous < 0:
                cursor[p] = i
            else:
                next_event[previous] = i
            last_event[p] = i

    for i in range(n):
        x, y = int(xs[i]), int(ys[i])
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        p_current = y * w + x
        if selection_mode == 1:
            cursor[p_current] = next_event[i]

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

                    if selection_mode == 1:
                        # Select the event with the smallest temporal distance.
                        dt = 1e18
                        if history_ts[p] > -1e17:
                            dt = t - history_ts[p]

                        j = cursor[p]
                        if j >= 0:
                            dt = min(dt, abs(ts[j] - t))

                        if dt >= 1e17:
                            continue
                    else:
                        # Use the latest processed event at the neighboring pixel.
                        if history_ts[p] > -1e17:
                            dt = t - history_ts[p]
                        else:
                            # Fall back to the first observed event at this pixel
                            # in the window when no window history is available.
                            # Only an existing event is used as a support candidate;
                            # pixels with no event in the window contribute nothing.
                            if first_ts[p] >= 1e17:
                                continue
                            dt = abs(first_ts[p] - t)

                    if weight_mode == 0:
                        weight = tau / (tau + dt)
                    else:
                        weight = max(0.0, 1.0 - dt / tau)

                    support += weight

            scores[i] = support

        history_ts[p_current] = t

    return scores


def spatial_support_ebe(
    events, feat, labels, pass_mask, resolution, cn,
    *, tau_us=4000.0, radius=2,
    weighting="hyperbolic", selection="nearest",
):
    """Map topology labels to events and compute temporal support."""
    if weighting not in ("hyperbolic", "linear"):
        raise ValueError("weighting must be hyperbolic or linear")
    if selection not in ("nearest", "history"):
        raise ValueError("selection must be nearest or history")
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

    if selection == "nearest":
        scores = compute_temporal_support(
            xs, ys, ts, c_ids, alive_mask,
            h, w, float(tau_us), int(radius), weight_mode,
        )
    else:
        scores = compute_temporal_support(
            xs, ys, ts, c_ids, alive_mask,
            h, w, float(tau_us), int(radius), weight_mode, 0,
        )

    out[valid_idx] = scores
    return out
