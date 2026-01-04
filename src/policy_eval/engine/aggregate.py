from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from policy_eval.engine.experiment import TrialResult


@dataclass(frozen=True)
class SummaryStats:
    n: int
    mean: float
    min: float
    max: float


def summarize(trials: list[TrialResult]) -> dict[tuple[str, str], SummaryStats]:
    """
    Returns {(policy_name, metric_name) -> SummaryStats}
    """
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)

    for tr in trials:
        for mr in tr.metrics:
            buckets[(tr.policy, mr.metric)].append(mr.value)

    out: dict[tuple[str, str], SummaryStats] = {}
    for key, vals in buckets.items():
        n = len(vals)
        mean = sum(vals) / n
        out[key] = SummaryStats(n=n, mean=mean, min=min(vals), max=max(vals))

    return out
