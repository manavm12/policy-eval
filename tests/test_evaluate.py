from __future__ import annotations

from typing import Any

from policy_eval.abstractions.policy import PolicyContext
from policy_eval.core.types import Event, RunConfig, Trajectory
from policy_eval.engine.evaluate import evaluate
from policy_eval.engine.pareto import MetricSpec


class Policy0:
    name = "p0"
    def decide(self, ctx: PolicyContext) -> int:
        return 0


class Policy1:
    name = "p1"
    def decide(self, ctx: PolicyContext) -> int:
        return 1


class TinyDomain:
    name = "tiny"

    def initial_state(self, rng: Any) -> int:
        return 0

    def actors(self, state: int):
        return []

    def policy_context(self, state: int, t: int) -> PolicyContext:
        return PolicyContext(t=t, system_view=state)

    def observe(self, state: int, actor, t: int):
        raise RuntimeError("no actors")

    def transition(self, state: int, policy_action: int, actor_actions, rng: Any, t: int) -> int:
        return state + int(policy_action)

    def record(self, state: int, t: int) -> Event:
        return Event(t=t, payload=state)

    def finalize(self, events, final_state) -> Trajectory:
        return Trajectory(events=tuple(events), final_state=final_state)


class FinalStateMetric:
    name = "final"
    def evaluate(self, traj: Trajectory) -> float:
        return float(traj.final_state)


def test_evaluate_returns_summaries_and_pareto():
    res = evaluate(
        domain=TinyDomain(),
        policies=[Policy0(), Policy1()],
        metrics=[FinalStateMetric()],
        seeds=[0, 1, 2],
        cfg=RunConfig(horizon=5),
        pareto_metrics=[MetricSpec(name="final", direction="higher_better")],
    )

    assert res.pareto is not None
    assert set(res.pareto) == {"p1"}  # always higher final state
    assert ("p0", "final") in res.summaries
    assert ("p1", "final") in res.summaries
