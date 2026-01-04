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
    """
    Run one simulation with a fixed domain, policy, and seed.
    """
    rng = RNG(seed).generator()

    # 1. Initialize world
    state: Any = domain.initial_state(rng)
    events = []

    # 2. Run time forward
    for t in range(cfg.horizon):
        # Domain advances the world
        state, event = domain.step(state, rng, t)
        events.append(event)

    # 3. Finalize trajectory
    traj = domain.finalize(events, state)

    return RunResult(
        domain=domain.name,
        policy=policy.name,
        seed=seed,
        trajectory=traj,
    )
