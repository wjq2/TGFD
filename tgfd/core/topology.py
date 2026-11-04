"""Pixel statistics and topology-prior generation using vectorized graph construction."""
from dataclasses import dataclass
from .graph import compute_edges, ew_pol_diff
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def extract_features(events, resolution):
    """Aggregate positive/negative counts into active pixel nodes."""
    W, H = resolution
    h, w = (H, W)
    if len(events) == 0:
        return None
    xs_all = events[:, 1].astype(int)
    ys_all = events[:, 2].astype(int)
    ps_all = events[:, 3]
    input_valid = (xs_all >= 0) & (xs_all < W) & (ys_all >= 0) & (ys_all < H)
    xs, ys, ps = (xs_all[input_valid], ys_all[input_valid], ps_all[input_valid])
    if len(xs) == 0:
        return None
    idx = ys * W + xs
    C_on = np.bincount(idx[ps == 1], minlength=h * w).reshape(h, w).astype(np.float32)
    C_off = np.bincount(idx[ps != 1], minlength=h * w).reshape(h, w).astype(np.float32)
    active = C_on + C_off > 0
    n_nodes = int(np.sum(active))
    if n_nodes == 0:
        return None
    id_map = np.full((h, w), -1, np.int32)
    id_map[active] = np.arange(n_nodes)
    N_total = C_on + C_off
    pol_map = np.where(N_total > 0, C_on / np.maximum(N_total, 1), 0.5)
    pad = ((1, 1), (1, 1))
    return {'input_valid': input_valid, 'xs': xs, 'ys': ys, 'n_nodes': n_nodes, 'id_map': id_map, 'id_p': np.pad(id_map, pad, constant_values=-1), 'mask_p': np.pad(active, pad), 'Con_p': np.pad(C_on, pad), 'Coff_p': np.pad(C_off, pad), 'Pol_p': np.pad(pol_map, pad)}

def thresh_confidence(w, k=1.0):
  
    return max(0.0, float(np.mean(w) + k * np.std(w)))

@dataclass
class TopologyPrior:
    features: dict
    labels: np.ndarray
    pass_mask: np.ndarray
    component_sizes: np.ndarray
    event_node_ids: np.ndarray


def build_topology_prior(events, resolution, *, alpha=1.0, min_component_size=2, eps=1e-5):
    
    feat = extract_features(events, resolution)
    if feat is None:
        return None
    fs, fe, fw = compute_edges(feat, eps=eps)
    if fs is None:
        return None
    threshold = thresh_confidence(fw, k=alpha)
    keep = fw <= threshold
    if keep.sum() == 0:
        return None
    graph = coo_matrix((fw[keep], (fs[keep], fe[keep])),
                       shape=(feat['n_nodes'], feat['n_nodes']))
    nc, labels = connected_components(graph, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=nc)
    pass_mask = sizes >= min_component_size
    event_nodes = feat['id_map'][feat['ys'], feat['xs']]
    valid = event_nodes != -1
    if not np.any(valid):
        return None
    return TopologyPrior(feat, labels, pass_mask, sizes, event_nodes[valid])
