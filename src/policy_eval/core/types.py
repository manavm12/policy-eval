from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Generic run settings (domain-agnostic)."""
    horizon: int


@dataclass(frozen=True)
class Event:
    """
    A standardized log entry produced by the domain each timestep.
    payload is domain-defined.
    """
    t: int
    payload: Any


@dataclass(frozen=True)
class Trajectory:
    """
    The ONLY thing metrics are allowed to inspect.
    Immutable by design.
    """
    events: tuple[Event, ...]
    final_state: Any
