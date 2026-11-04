"""Vectorized construction of an undirected 8-neighborhood pixel graph.

Padded array slices align neighboring pixels. Boolean masks select active
pairs, and edge costs are evaluated in batches. Only four directions are
enumerated, so each undirected edge occurs once. No GPU or explicit
multithreading is assumed.
"""
import numpy as np

def ew_pol_diff(feat, conn, sc, sl, *, eps=1e-05):
    """Compute polarity-aware edge costs from pixel event statistics."""
    p = feat['Con_p'][sc][conn]
    q = feat['Con_p'][sl][conn]
    op = feat['Coff_p'][sc][conn]
    oq = feat['Coff_p'][sl][conn]
    Np = p + op
    Nq = q + oq
    pp = feat['Pol_p'][sc][conn]
    pq = feat['Pol_p'][sl][conn]
    R = np.abs(p - q) + np.abs(op - oq)
    denom = np.sqrt(Np * Nq + eps) + eps
    return (R + 1 + np.abs(pp - pq)) / denom

def compute_edges(feat, *, eps=1e-05):
    """Enumerate four directions of the undirected 8-neighborhood."""
    mask_p = feat['mask_p']
    id_p = feat['id_p']
    sc = (slice(1, -1), slice(1, -1))
    dirs = [(slice(1, -1), slice(2, None)), (slice(2, None), slice(1, -1)), (slice(2, None), slice(2, None)), (slice(2, None), slice(0, -2))]
    all_s, all_e, all_w = ([], [], [])
    for sl in dirs:
        conn = mask_p[sc] & mask_p[sl]
        if not np.any(conn):
            continue
        wts = ew_pol_diff(feat, conn, sc, sl, eps=eps)
        all_s.append(id_p[sc][conn])
        all_e.append(id_p[sl][conn])
        all_w.append(wts)
    if not all_s:
        return (None, None, None)
    return (np.concatenate(all_s), np.concatenate(all_e), np.concatenate(all_w))

