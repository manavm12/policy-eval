from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from policy_eval.engine.compare import DistSummary


Direction = Literal["higher_better", "lower_better"]
Criterion = Literal["mean", "p50", "p90", "min"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: Direction
    criterion: Criterion = "mean"
    tol: float = 0.0


def _value(s: DistSummary, criterion: Criterion) -> float:
    return float(getattr(s, criterion))


def _better_or_equal(a: float, b: float, direction: Direction, tol: float) -> bool:
    if direction == "higher_better":
        return a >= b - tol
    return a <= b + tol


def _strictly_better(a: float, b: float, direction: Direction, tol: float) -> bool:
    if direction == "higher_better":
        return a > b + tol
    return a < b - tol


def pareto_front(
    summaries: dict[tuple[str, str], DistSummary],
    metrics: list[MetricSpec],
) -> list[str]:
    """
    summaries: {(policy, metric_name) -> DistSummary}
    returns: list of policy names that are Pareto-optimal across the given metrics.
    """
    policies = sorted({p for (p, _) in summaries.keys()})

    def dominates(pa: str, pb: str) -> bool:
        # pa dominates pb if pa is >= on all metrics and strictly better on at least one
        any_strict = False
        for ms in metrics:
            sa = summaries[(pa, ms.name)]
            sb = summaries[(pb, ms.name)]
            va = _value(sa, ms.criterion)
            vb = _value(sb, ms.criterion)

            if not _better_or_equal(va, vb, ms.direction, ms.tol):
                return False
            if _strictly_better(va, vb, ms.direction, ms.tol):
                any_strict = True
        return any_strict

    front = []
    for p in policies:
        dominated = False
        for q in policies:
            if p == q:
                continue
            if dominates(q, p):
                dominated = True
                break
        if not dominated:
            front.append(p)

    return front
