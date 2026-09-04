from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models import AlgorithmPlan, CapabilitySpec


@dataclass
class SearchTrace:
    strategy: str
    beam_width: int
    expanded: int
    selected: list[str]
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "beam_width": self.beam_width, "expanded": self.expanded, "selected": self.selected, "scores": self.scores}


class BeamSearchPlanner:
    """A lightweight beam search over algorithm plans and business preferences."""

    def search(self, plans: list[AlgorithmPlan], spec: CapabilitySpec, beam_width: int = 3) -> tuple[list[AlgorithmPlan], SearchTrace]:
        def score(plan: AlgorithmPlan) -> float:
            historical = plan.expected_metrics.get("roc_auc", 0.5)
            interpretability = 0.15 if "prefer_interpretable" in spec.constraints and plan.algorithm_name == "Logistic Regression" else 0.0
            latency = 0.08 if "low_latency" in spec.constraints and plan.algorithm_name == "Logistic Regression" else 0.0
            resource_penalty = {"Random Forest": 0.02, "Gradient Boosting": 0.01}.get(plan.algorithm_name, 0.0)
            return float(historical + interpretability + latency - resource_penalty)

        scores = {plan.algorithm_id: score(plan) for plan in plans}
        ranked = sorted(plans, key=lambda plan: scores[plan.algorithm_id], reverse=True)
        selected = ranked[: max(1, beam_width)]
        trace = SearchTrace("beam_search", max(1, beam_width), len(plans), [p.algorithm_id for p in selected], {k: round(v, 6) for k, v in scores.items()})
        return selected, trace

