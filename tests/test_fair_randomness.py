from __future__ import annotations

from typing import Any

from policy_eval.abstractions.actor import Observation
from policy_eval.abstractions.policy import PolicyContext
from policy_eval.core.types import Event, RunConfig, Trajectory
from policy_eval.engine.simulator import simulate


class RandomActor:
    id = "a"

    def act(self, obs: Observation, rng: Any) -> int:
        # actor randomness should be identical across policies for same seed
        return int(rng.integers(0, 3))  # 0,1,2


class PolicyA:
    name = "A"
    def decide(self, ctx: PolicyContext) -> int:
        return 0  # triggers extra RNG use in domain


class PolicyB:
    name = "B"
    def decide(self, ctx: PolicyContext) -> int:
        return 1  # does not trigger extra RNG use


class BranchyDomain:
    """
    If policy_action==0, transition consumes extra RNG.
    This should NOT affect actor randomness if simulator uses RNG substreams.
    """
    name = "branchy"

    def initial_state(self, rng: Any) -> dict[str, Any]:
        return {"x": 0, "last_actor": None}

    def actors(self, state):
        return [RandomActor()]

    def policy_context(self, state, t: int) -> PolicyContext:
        return PolicyContext(t=t, system_view={"x": state["x"]})

    def observe(self, state, actor, t: int) -> Observation:
        return Observation(t=t, data={"x": state["x"]})

    def transition(self, state, policy_action, actor_actions, rng: Any, t: int):
        # Policy-dependent RNG consumption (this is the trap)
        if policy_action == 0:
            _ = rng.integers(0, 10, size=1000)  # burn randomness

        a = int(actor_actions[0])
        return {"x": state["x"] + policy_action, "last_actor": a}

    def record(self, state, t: int) -> Event:
        return Event(t=t, payload={"x": state["x"], "last_actor": state["last_actor"]})

    def finalize(self, events, final_state) -> Trajectory:
        return Trajectory(events=tuple(events), final_state=final_state)


def test_actor_randomness_same_across_policies_with_same_seed():
    domain = BranchyDomain()
    cfg = RunConfig(horizon=20)
    seed = 123

    rA = simulate(domain, PolicyA(), cfg, seed=seed)
    rB = simulate(domain, PolicyB(), cfg, seed=seed)

    # Actor draws should match exactly across policies
    acts_A = [e.payload["last_actor"] for e in rA.trajectory.events]
    acts_B = [e.payload["last_actor"] for e in rB.trajectory.events]
    assert acts_A == acts_B

    # But system outcome can differ because policy differs
    assert rA.trajectory.final_state["x"] != rB.trajectory.final_state["x"]
