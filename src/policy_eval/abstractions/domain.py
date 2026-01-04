from __future__ import annotations
from typing import Any, Protocol

from policy_eval.abstractions.actor import Actor, Observation
from policy_eval.abstractions.policy import PolicyContext
from policy_eval.core.types import Event, Trajectory


class Domain(Protocol):
    """
    Defines the world being simulated.
    """

    name: str

    def initial_state(self, rng: Any) -> Any:
        ...

    def actors(self, state: Any) -> list[Actor]:
        ...

    def policy_context(self, state: Any, t: int) -> PolicyContext:
        ...

    def observe(self, state: Any, actor: Actor, t: int) -> Observation:
        ...

    def transition(
        self,
        state: Any,
        policy_action: Any,
        actor_actions: list[Any],
        rng: Any,
        t: int,
    ) -> Any:
        ...

    def record(self, state: Any, t: int) -> Event:
        ...

    def finalize(self, events: list[Event], final_state: Any) -> Trajectory:
        ...
