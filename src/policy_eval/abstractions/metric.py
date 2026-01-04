from __future__ import annotations
from typing import Protocol

from policy_eval.core.types import Trajectory


class Metric(Protocol):
    """
    Evaluates outcomes after a run.
    """

    name: str

    def evaluate(self, traj: Trajectory) -> float:
        """
        Pure function of the trajectory.
        Must not mutate anything.
        """
        ...
