"""TGFD algorithm stages and numerical configuration."""
from .config import Config
from .windowing import twminev_slices as aow_slices, fti_slices, fec_slices
from .topology import extract_features, compute_edges, build_topology_prior
