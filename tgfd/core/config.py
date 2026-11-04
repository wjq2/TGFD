"""Load and validate algorithm parameters."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    width: int
    height: int
    dt_ms: float
    n_min: int
    alpha: float
    min_component_size: int
    radius: int
    tau_us: float
    eps: float
    windowing: str
    n_events: int

    def __post_init__(self):
        if self.windowing not in ("aow", "fti", "fec"):
            raise ValueError("windowing must be aow, fti or fec")

        for key in (
            "width", "height", "n_min", "n_events",
            "min_component_size", "radius",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")

        for key in ("dt_ms", "tau_us", "eps"):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")

        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)
