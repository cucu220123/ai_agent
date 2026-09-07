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

        expanded_plans: list[AlgorithmPlan] = []
        preprocessing_variants = ["default", "balanced", "robust_missing"] if spec.task_type == "binary_classification" else ["default"]
        config_variants = ["baseline", "regularized", "resource_light"]
        for plan in plans:
            for pre in preprocessing_variants:
                for config in config_variants:
                    params = dict(plan.hyperparameters)
                    if pre == "balanced" and plan.algorithm_name in {"Logistic Regression", "Random Forest"}:
                        params["class_weight"] = "balanced"
                    if config == "resource_light":
                        params["n_estimators"] = min(int(params.get("n_estimators", 120)), 80)
                    expanded_plans.append(AlgorithmPlan(plan.algorithm_id + "__" + pre + "__" + config, plan.algorithm_name, plan.rationale, plan.preprocessing, params, plan.expected_metrics, plan.priority, plan.algorithm_id, pre, config, 0.0, plan.evidence_ids))
        def state_score(plan: AlgorithmPlan) -> float:
            base = score(plan)
            bonus = 0.03 if plan.preprocessing_variant == "balanced" and spec.class_imbalance else 0.0
            cost = 0.01 if plan.config_variant == "resource_light" else 0.0
            return base + bonus + cost
        scores = {plan.algorithm_id: state_score(plan) for plan in expanded_plans}
        ranked = sorted(expanded_plans, key=lambda plan: scores[plan.algorithm_id], reverse=True)
        selected: list[AlgorithmPlan] = []
        seen_bases: set[str] = set()
        for candidate in ranked:
            base = candidate.base_algorithm_id or candidate.algorithm_id
            if base not in seen_bases or len(selected) >= beam_width:
                selected.append(candidate)
                seen_bases.add(base)
            if len(selected) >= max(1, beam_width):
                break
        trace = SearchTrace("beam_search", max(1, beam_width), len(expanded_plans), [p.algorithm_id for p in selected], {k: round(v, 6) for k, v in scores.items()})
        return selected, trace
