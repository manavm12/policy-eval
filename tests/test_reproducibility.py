from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from policy_eval.abstractions.domain import Domain
from policy_eval.abstractions.policy import Policy, PolicyContext
from policy_eval.core.types import Event, RunConfig, Trajectory
from policy_eval.engine.simulator import simulate


# --- minimal deterministic domain for testing ---
class DeterministicDomain:
    name = "deterministic"

    def initial_state(self, rng: Any) -> int:
        return 0

    def actors(self, state: int):
        return []  # no actors needed for this test

    def policy_context(self, state: int, t: int) -> PolicyContext:
        return PolicyContext(t=t, system_view=state)

    def observe(self, state: int, actor, t: int):
        raise RuntimeError("No actors in this domain")

    def transition(self, state: int, policy_action: Any, actor_actions: list[Any], rng: Any, t: int) -> int:
        # policy_action is a deterministic increment
        return state + int(policy_action)

    def record(self, state: int, t: int) -> Event:
        return Event(t=t, payload=state)

    def finalize(self, events: list[Event], final_state: int) -> Trajectory:
        return Trajectory(events=tuple(events), final_state=final_state)


class IncOnePolicy:
    name = "inc_one"

    def decide(self, ctx: PolicyContext) -> int:
        return 1


def test_same_seed_same_result():
    domain = DeterministicDomain()
    policy = IncOnePolicy()
    cfg = RunConfig(horizon=10)

    r1 = simulate(domain, policy, cfg, seed=123)
    r2 = simulate(domain, policy, cfg, seed=123)

    assert r1.trajectory == r2.trajectory
