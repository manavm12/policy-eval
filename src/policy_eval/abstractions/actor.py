from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Observation:
    """
    What an actor can see.
    Domain-defined, policy-blind.
    """
    t: int
    data: Any


class Actor(Protocol):
    """
    An entity that reacts to observations.
    """

    id: str

    def act(self, obs: Observation, rng: Any) -> Any:
        """
        Given an observation, return an action.
        Action type is domain-defined.
        """
        ...
