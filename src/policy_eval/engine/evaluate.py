from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from policy_eval.abstractions.domain import Domain
from policy_eval.abstractions.metric import Metric
from policy_eval.abstractions.policy import Policy
from policy_eval.core.types import RunConfig
from policy_eval.engine.compare import DistSummary, summarize_distributions
from policy_eval.engine.experiment import TrialResult, run_experiment
from policy_eval.engine.pareto import MetricSpec, pareto_front


@dataclass(frozen=True)
class EvaluationResult:
    trials: list[TrialResult]
    summaries: dict[tuple[str, str], DistSummary]
    pareto: list[str] | None


def evaluate(
    *,
    domain: Domain,
    policies: Iterable[Policy],
    metrics: Iterable[Metric],
    seeds: Iterable[int],
    cfg: RunConfig,
    pareto_metrics: list[MetricSpec] | None = None,
) -> EvaluationResult:
    """
    One-stop API:
    - runs trials
    - summarizes distributions
    - optionally computes Pareto front across multiple metrics
    """
    trials = run_experiment(
        domain=domain,
        policies=policies,
        metrics=metrics,
        seeds=seeds,
        cfg=cfg,
    )
    summaries = summarize_distributions(trials)

    pareto_set = None
    if pareto_metrics is not None:
        pareto_set = pareto_front(summaries, pareto_metrics)

    return EvaluationResult(trials=trials, summaries=summaries, pareto=pareto_set)
