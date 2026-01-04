from __future__ import annotations

from typing import Any

from policy_eval.abstractions.actor import Observation
from policy_eval.abstractions.policy import PolicyContext
from policy_eval.core.types import Event, RunConfig, Trajectory
from policy_eval.engine.simulator import simulate


class CountingActor:
    """
    Actor that only reacts to what it observes.
    If policy leaked into observation, we could detect it.
    """
    id = "a"

    def act(self, obs: Observation, rng: Any) -> int:
        # If policy leaked into obs.data, this would change.
        assert "policy" not in (obs.data or {})
        return 1  # deterministic action


class PolicyIncOne:
    name = "inc_1"

    def decide(self, ctx: PolicyContext) -> int:
        return 1


class PolicyIncTwo:
    name = "inc_2"

    def decide(self, ctx: PolicyContext) -> int:
        return 2


class IsolationDomain:
    """
    World state is a single integer.
    Policy chooses how much to increment state each step.
    Actor always returns 1, but actor action is ignored here.
    Crucially: observe() must not include policy info.
    """
    name = "isolation"

    def initial_state(self, rng: Any) -> int:
        return 0

    def actors(self, state: int):
        return [CountingActor()]

    def policy_context(self, state: int, t: int) -> PolicyContext:
        return PolicyContext(t=t, system_view={"state": state})

    def observe(self, state: int, actor: CountingActor, t: int) -> Observation:
        # Policy must NOT leak in here.
        return Observation(t=t, data={"state": state})

    def transition(
        self,
        state: int,
        policy_action: int,
        actor_actions: list[Any],
        rng: Any,
        t: int,
    ) -> int:
        return state + int(policy_action)

    def record(self, state: int, t: int) -> Event:
        return Event(t=t, payload=state)

    def finalize(self, events: list[Event], final_state: int) -> Trajectory:
        return Trajectory(events=tuple(events), final_state=final_state)


def test_policy_does_not_leak_to_actor_observations():
    domain = IsolationDomain()
    cfg = RunConfig(horizon=5)

    r1 = simulate(domain, PolicyIncOne(), cfg, seed=0)
    r2 = simulate(domain, PolicyIncTwo(), cfg, seed=0)

    # Same actor, same seed, only policy differs.
    # Actor is policy-blind; only domain transition applies policy_action.
    assert r1.trajectory.final_state == 5
    assert r2.trajectory.final_state == 10
