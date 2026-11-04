"""AOW, FTI and FEC window strategies for event streams."""
import numpy as np

def fti_slices(events, dt_ms):
    dt = dt_ms * 1000
    ts = events[:, 0]
    n = len(ts)
    if n == 0:
        return []
    slices = []
    t = ts[0]
    while t <= ts[-1]:
        s = int(np.searchsorted(ts, t, 'left'))
        e = int(np.searchsorted(ts, t + dt, 'right'))
        e = min(e, n)
        if e > s:
            slices.append((s, e))
        t += dt
    return slices

def fec_slices(events, n_ev):
    n = len(events)
    if n == 0:
        return []
    slices = []
    for s in range(0, n, n_ev):
        e = min(s + n_ev, n)
        if e > s:
            slices.append((s, e))
    return slices

def twminev_slices(events, dt_ms, n_min):
    dt = dt_ms * 1000
    ts = events[:, 0]
    n = len(ts)
    if n == 0:
        return []
    slices = []
    t = ts[0]
    while t <= ts[-1]:
        s = int(np.searchsorted(ts, t, 'left'))
        e = int(np.searchsorted(ts, t + dt, 'right'))
        e = min(e, n)
        if e - s < n_min:
            new_s = max(0, e - n_min)
            if e > new_s:
                slices.append((new_s, e))
        else:
            slices.append((s, e))
        t += dt
    return slices

aow_slices = twminev_slices


def make_windows(events, config):
    """Dispatch a configured strategy; AOW is the TGFD default."""
    if config.windowing == 'aow':
        return aow_slices(events, config.dt_ms, config.n_min)
    if config.windowing == 'fti':
        return fti_slices(events, config.dt_ms)
    if config.windowing == 'fec':
        return fec_slices(events, config.n_events)
    raise ValueError("windowing must be 'aow', 'fti' or 'fec'")
