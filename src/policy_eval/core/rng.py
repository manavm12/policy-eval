from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class RNG:
    """
    Reproducible RNG wrapper.

    Key idea: allow deterministic substreams so randomness is invariant to
    call order and policy-dependent branching.
    """
    seed: int

    def generator(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def fork(self, *keys: object) -> np.random.Generator:
        """
        Create a deterministic sub-RNG from (seed, keys...).
        Same seed+keys => same stream. Policy name is NOT included.
        """
        h = hashlib.blake2b(digest_size=8)
        h.update(str(self.seed).encode("utf-8"))
        for k in keys:
            h.update(b"|")
            h.update(str(k).encode("utf-8"))
        sub_seed = int.from_bytes(h.digest(), "big", signed=False)
        return np.random.default_rng(sub_seed)
