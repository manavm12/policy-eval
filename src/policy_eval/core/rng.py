from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RNG:
    """
    Reproducible RNG wrapper.
    A numpy Generator seeded per run.
    """

    seed: int

    def generator(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)
