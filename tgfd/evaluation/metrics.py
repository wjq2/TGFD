
import numpy as np


def compute_esr(events, resolution, M=None):
    """Evaluate one block; default M=floor(2*N/3) follows E-MLB.

    Events have canonical columns [t_us, x, y, p]. M is an optional explicit
    reference-count override for older experiment configurations. None uses
    the original E-MLB rule. Fewer than two events return None.
    """
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] < 3:
        raise ValueError('events must be a 2D array with t, x, y columns')
    W, H = resolution
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
           for v in (W, H)):
        raise ValueError('resolution must contain positive integer width and height')
    xy = events[:, 1:3]
    if (not np.all(np.isfinite(xy)) or np.any(xy != np.floor(xy)) or
            np.any((xy[:, 0] < 0) | (xy[:, 0] >= W) |
                   (xy[:, 1] < 0) | (xy[:, 1] >= H))):
        raise ValueError('ESR requires valid integer pixel coordinates')
    N = len(events)
    if N < 2:
        return None
    M = int(N * 2 / 3) if M is None else M
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)) or not 0 < M < N:
        raise ValueError('reference count M must be an integer between 0 and N')
    xs, ys = xy.astype(np.int64).T
    n = np.bincount(ys * W + xs, minlength=W * H).astype(np.float64)
    K = W * H
    ntss = (n * (n - 1)).sum() / (N + np.spacing(1)) / (N - 1 + np.spacing(1))
    ln = K - ((1 - M / N) ** n).sum()
    return float(np.sqrt(ntss * ln))


def compute_mesr(events, resolution, N=30000, M=None):
    """Average ESR over complete non-overlapping N-event blocks.

    Default block size is the E-MLB benchmark's 30000. The trailing
    incomplete block is omitted; no complete block returns None, rather
    than a substitute raw-stream score. Input is expected in time order.
    """
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError('block size N must be an integer >= 2')
    if M is not None and (isinstance(M, bool) or not isinstance(M, (int, np.integer))
                          or not 0 < M < N):
        raise ValueError('reference count M must be an integer between 0 and N')
    values = [compute_esr(events[start:start + N], resolution, M)
              for start in range(0, len(events) - N + 1, N)]
    return float(np.mean(values)) if values else None
