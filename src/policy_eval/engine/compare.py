from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from policy_eval.engine.experiment import TrialResult


Direction = Literal["higher_better", "lower_better"]


@dataclass(frozen=True)
class DistSummary:
    n: int
    mean: float
    std: float
    min: float
    p10: float
    p50: float
    p90: float
    max: float


def _quantile(sorted_vals: list[float], q: float) -> float:
    """
    Simple linear interpolation quantile.
    q in [0, 1]
    """
    if not sorted_vals:
        raise ValueError("Cannot compute quantile of empty list")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def summarize_distributions(
    trials: Iterable[TrialResult],
) -> dict[tuple[str, str], DistSummary]:
    """
    Returns {(policy_name, metric_name) -> distribution summary}.
    """
    buckets: dict[tuple[str, str], list[float]] = {}

    for tr in trials:
        for mr in tr.metrics:
            key = (tr.policy, mr.metric)
            buckets.setdefault(key, []).append(float(mr.value))

    out: dict[tuple[str, str], DistSummary] = {}
    for key, vals in buckets.items():
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean = sum(vals_sorted) / n
        var = sum((x - mean) ** 2 for x in vals_sorted) / n
        std = var**0.5

        out[key] = DistSummary(
            n=n,
            mean=mean,
            std=std,
            min=vals_sorted[0],
            p10=_quantile(vals_sorted, 0.10),
            p50=_quantile(vals_sorted, 0.50),
            p90=_quantile(vals_sorted, 0.90),
            max=vals_sorted[-1],
        )
    return out


def dominates(
    a: DistSummary,
    b: DistSummary,
    *,
    direction: Direction,
    criterion: Literal["mean", "p50", "p90", "min"] = "mean",
    tol: float = 0.0,
) -> bool:
    """
    Returns True if A "dominates" B under a chosen scalar criterion and direction.

    tol: allows a small slack (e.g. tol=1e-6).
    """
    va = getattr(a, criterion)
    vb = getattr(b, criterion)

    if direction == "higher_better":
        return va >= vb + tol
    else:
        return va <= vb - tol


def pairwise_dominance_report(
    summaries: dict[tuple[str, str], DistSummary],
    *,
    metric: str,
    direction: Direction,
    criterion: Literal["mean", "p50", "p90", "min"] = "mean",
    tol: float = 0.0,
) -> list[tuple[str, str]]:
    """
    Returns list of (winner_policy, loser_policy) pairs for the given metric.
    """
    # collect policies that have this metric
    policies = sorted({p for (p, m) in summaries.keys() if m == metric})
    wins: list[tuple[str, str]] = []

    for i in range(len(policies)):
        for j in range(len(policies)):
            if i == j:
                continue
            pa, pb = policies[i], policies[j]
            a = summaries[(pa, metric)]
            b = summaries[(pb, metric)]
            if dominates(a, b, direction=direction, criterion=criterion, tol=tol):
                wins.append((pa, pb))

    return wins
