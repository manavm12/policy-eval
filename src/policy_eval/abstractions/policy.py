from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PolicyContext:
    """
    What the policy is allowed to see.
    Domain-defined, actor-blind.
    """
    t: int
    system_view: Any


class Policy(Protocol):
    """
    System-level rules.
    """

    name: str

    def decide(self, ctx: PolicyContext) -> Any:
        """
        Given system context, return a policy action.
        Action type is domain-defined.
        """
        ...
