from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from policy_eval.abstractions.domain import Domain
from policy_eval.abstractions.metric import Metric
from policy_eval.abstractions.policy import Policy
from policy_eval.core.types import RunConfig
from policy_eval.engine.simulator import RunResult, simulate


@dataclass(frozen=True)
class MetricResult:
    metric: str
    value: float


@dataclass(frozen=True)
class TrialResult:
    domain: str
    policy: str
    seed: int
    metrics: tuple[MetricResult, ...]
    run: RunResult


def run_experiment(
    *,
    domain: Domain,
    policies: Iterable[Policy],
    metrics: Iterable[Metric],
    seeds: Iterable[int],
    cfg: RunConfig,
) -> list[TrialResult]:
    """
    Run many seeds across many policies.
    Returns per-(policy, seed) results so you can compute distributions later.
    """
    metrics_list = list(metrics)
    results: list[TrialResult] = []

    for policy in policies:
        for seed in seeds:
            run = simulate(domain, policy, cfg, seed=seed)
            scored = tuple(
                MetricResult(metric=m.name, value=float(m.evaluate(run.trajectory)))
                for m in metrics_list
            )
            results.append(
                TrialResult(
                    domain=run.domain,
                    policy=run.policy,
                    seed=seed,
                    metrics=scored,
                    run=run,
                )
            )

    return results
