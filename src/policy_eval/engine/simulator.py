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


def simulate(
    domain: Domain,
    policy: Policy,
    cfg: RunConfig,
    seed: int,
) -> RunResult:
    rng = RNG(seed).generator()

    state: Any = domain.initial_state(rng)
    events = []

    for t in range(cfg.horizon):
        # 1. Policy acts (system-level rules)
        ctx = domain.policy_context(state, t)
        policy_action = policy.decide(ctx)

        # 2. Actors act (behavior)
        actor_actions = []
        for actor in domain.actors(state):
            obs = domain.observe(state, actor, t)
            action = actor.act(obs, rng)
            actor_actions.append(action)

        # 3. World transitions
        state = domain.transition(
            state=state,
            policy_action=policy_action,
            actor_actions=actor_actions,
            rng=rng,
            t=t,
        )

        # 4. Record what happened
        events.append(domain.record(state, t))

    traj = domain.finalize(events, state)

    return RunResult(
        domain=domain.name,
        policy=policy.name,
        seed=seed,
        trajectory=traj,
    )
