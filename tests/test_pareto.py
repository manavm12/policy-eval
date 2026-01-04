from __future__ import annotations

from policy_eval.engine.compare import DistSummary
from policy_eval.engine.pareto import MetricSpec, pareto_front


def test_pareto_front_basic():
    # Two metrics: efficiency (higher better) and fairness (higher better)
    # Policy A: high efficiency, low fairness
    # Policy B: low efficiency, high fairness
    # Policy C: worse than A on both => should be dominated
    summaries = {
        ("A", "eff"): DistSummary(n=10, mean=10, std=0, min=10, p10=10, p50=10, p90=10, max=10),
        ("A", "fair"): DistSummary(n=10, mean=1, std=0, min=1, p10=1, p50=1, p90=1, max=1),

        ("B", "eff"): DistSummary(n=10, mean=1, std=0, min=1, p10=1, p50=1, p90=1, max=1),
        ("B", "fair"): DistSummary(n=10, mean=10, std=0, min=10, p10=10, p50=10, p90=10, max=10),

        ("C", "eff"): DistSummary(n=10, mean=1, std=0, min=1, p10=1, p50=1, p90=1, max=1),
        ("C", "fair"): DistSummary(n=10, mean=1, std=0, min=1, p10=1, p50=1, p90=1, max=1),
    }

    metrics = [
        MetricSpec(name="eff", direction="higher_better", criterion="mean"),
        MetricSpec(name="fair", direction="higher_better", criterion="mean"),
    ]

    front = pareto_front(summaries, metrics)
    assert set(front) == {"A", "B"}
