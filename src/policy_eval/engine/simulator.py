from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from policy_eval.abstractions.domain import Domain
from policy_eval.abstractions.actor import Actor, Observation
from policy_eval.abstractions.policy import Policy
from policy_eval.core.rng import RNG
from policy_eval.core.types import RunConfig, Trajectory


@dataclass(frozen=True)
class RunResult:
    """
    Result of a single simulation run.
    """
    domain: str
    policy: str
    seed: int
    trajectory: Trajectory


def simulate(domain: Domain, policy: Policy, cfg: RunConfig, seed: int) -> RunResult:
    base = RNG(seed)

    state: Any = domain.initial_state(base.fork("init"))
    events = []

    for t in range(cfg.horizon):
        # Policy (no rng needed right now)
        ctx = domain.policy_context(state, t)
        policy_action = policy.decide(ctx)

        # Actors get their own deterministic streams
        actor_actions = []
        for actor in domain.actors(state):
            obs = domain.observe(state, actor, t)
            actor_rng = base.fork("actor", actor.id, t)
            actor_actions.append(actor.act(obs, actor_rng))

        # Domain transition gets its own stream
        trans_rng = base.fork("transition", t)
        state = domain.transition(
            state=state,
            policy_action=policy_action,
            actor_actions=actor_actions,
            rng=trans_rng,
            t=t,
        )

        events.append(domain.record(state, t))

    traj = domain.finalize(events, state)
    return RunResult(domain=domain.name, policy=policy.name, seed=seed, trajectory=traj)
