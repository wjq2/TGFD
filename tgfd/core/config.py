"""Algorithm configuration, independent of paths and datasets."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Config:
    width: int = 346
    height: int = 260
    dt_ms: float = 16.0
    n_min: int = 15000
    alpha: float = 1.0
    min_component_size: int = 2
    radius: int = 2
    tau_us: float = 4000.0
    eps: float = 1e-05
    windowing: str = "aow"
    n_events: int = 15000

    def __post_init__(self):
        if self.windowing not in ("aow", "fti", "fec"):
            raise ValueError("windowing must be aow, fti or fec")
        for key in ('width', 'height', 'n_min', 'n_events', 'min_component_size', 'radius'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('dt_ms', 'tau_us', 'eps'):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    def to_dict(self):
        return asdict(self)
