"""Temporal support for topology-guided event denoising."""
import numpy as np
from numba import njit

@njit(cache=True)
def compute_history_first_future(xs, ys, ts, c_ids, alive_mask, h, w, tau, pw):
    """Use the latest processed neighbor event, or its first window event if absent."""
    n = len(xs)
    scores = np.zeros(n, dtype=np.float32)
    first_ts = np.full(h * w, 1e+18, dtype=np.float64)
    history_ts = np.full(h * w, -1e+18, dtype=np.float64)
    cid_map = np.full(h * w, -1, dtype=np.int32)
    for i in range(n):
        x, y = (int(xs[i]), int(ys[i]))
        if x < 0 or x >= w or y < 0 or (y >= h):
            continue
        p = y * w + x
        cid_map[p] = c_ids[i]
        if ts[i] < first_ts[p]:
            first_ts[p] = ts[i]
    for i in range(n):
        x, y = (int(xs[i]), int(ys[i]))
        if x < 0 or x >= w or y < 0 or (y >= h):
            continue
        t, cid = (ts[i], c_ids[i])
        if alive_mask[i]:
            s = 0.0
            for dy in range(-pw, pw + 1):
                for dx in range(-pw, pw + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= w or ny < 0 or (ny >= h):
                        continue
                    p = ny * w + nx
                    if cid_map[p] != cid:
                        continue
                    tj_hist = history_ts[p]
                    if tj_hist > -1e+17:
                        dt = t - tj_hist
                    else:
                        tj_future = first_ts[p]
                        if tj_future >= 1e+17:
                            continue
                        dt = abs(tj_future - t)
                    s += tau / (tau + dt)
            scores[i] = s
        history_ts[y * w + x] = t
    return scores

def spatial_support_ebe(events, feat, labels, pass_mask, resolution, cn, *, tau_us=4000.0, radius=2):
    """Map topology labels to events and compute temporal support scores."""
    out = np.zeros(len(events), np.float32)
    h, w = (resolution[1], resolution[0])
    input_valid = feat['input_valid']
    valid_idx = np.where(input_valid)[0]
    if len(valid_idx) == 0:
        return out
    xs = feat['xs'].astype(np.int32)
    ys = feat['ys'].astype(np.int32)
    ts = events[input_valid, 0].astype(np.float64)
    c_ids = labels[cn].astype(np.int32)
    alive_mask = pass_mask[c_ids]
    tau = float(tau_us)
    pw = int(radius)
    ev_scores = compute_history_first_future(xs, ys, ts, c_ids, alive_mask, h, w, tau, pw)
    out[valid_idx] = ev_scores
    return out
