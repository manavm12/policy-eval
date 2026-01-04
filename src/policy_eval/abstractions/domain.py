from __future__ import annotations
from typing import Any, Protocol

from policy_eval.core.types import Event, Trajectory


class Domain(Protocol):
    """
    Defines the world being simulated.
    """

    name: str

    def initial_state(self, rng: Any) -> Any:
        """Create the starting state of the world."""
        ...

    def step(self, state: Any, rng: Any, t: int) -> tuple[Any, Event]:
        """
        Advance the world by one timestep.
        Returns the new state and an event to record.
        """
        ...

    def finalize(self, events: list[Event], final_state: Any) -> Trajectory:
        """Package the run into a Trajectory."""
        ...
