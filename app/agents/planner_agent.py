"""Combine validated LLM proposals with contextual priors and deterministic constraints."""
from __future__ import annotations
import math
from typing import Any
from app.models import AlgorithmPlan, CapabilitySpec, KnowledgeContext
from app.experience.retriever import ExperienceRetriever
from app.plugins.registry import DEFAULT_REGISTRY
from app.metrics.registry import METRIC_REGISTRY


class PlannerAgent:
    def run(self, spec: CapabilitySpec, knowledge: KnowledgeContext) -> list[AlgorithmPlan]:
        plugins = {p.id: p for p in DEFAULT_REGISTRY.algorithms_for(spec.task_type)}
        names = [name for name in (spec.candidate_algorithms or list(plugins)) if name in plugins and name != "dummy_classifier"]
        if "dummy_classifier" in spec.candidate_algorithms: names.append("dummy_classifier")
        priors = ExperienceRetriever().algorithm_priors(spec, knowledge.historical_cases)
        primary = METRIC_REGISTRY.primary(spec.task_type, spec.metrics, bool(spec.target_column))
        seed = {a["id"]: a.get("historical_metrics", {}) for a in knowledge.algorithms}
        advice = knowledge.planner_advice
        plans = []
        for key in dict.fromkeys(names):
            plugin = plugins[key]
            algorithm_id = f"algorithm_{key}"
            prior = priors.get(algorithm_id, {})
            measured = prior.get("historical_score")
            cold = seed.get(algorithm_id, {}).get(primary)
            expected = {primary: measured} if measured is not None else ({primary: cold} if cold is not None else {})
            if measured is None:
                utility = .5 + .1 * ((float(cold) if cold is not None else .5) - .5)
            else:
                utility = self._utility(primary, float(measured), spec)
                confidence = float(prior.get("effective_weight", 0)) / (1 + float(prior.get("effective_weight", 0)))
                utility = .5 + confidence * (utility - .5)
            components: dict[str, Any] = {
                "performance_prior": utility, "success": .10 * prior.get("success_rate", .5),
                "stability": .04 * prior.get("stability_rate", .5),
                "exploration": .12 * prior.get("exploration_bonus", 1.0),
                "runtime_cost": -.01 * math.log1p(prior.get("mean_runtime", 0)),
                "interpretability": .22 if spec.interpretability_requirement == "high" and plugin.resource_profile == "low" else 0,
                "resource_fit": -.12 if spec.resource_constraints.get("prefer_low_memory") and plugin.resource_profile != "low" else 0,
            }
            params = dict(plugin.default_params)
            recommendations = advice.get("hyperparameter_recommendations", {}).get(key, {})
            allowed_keys = set(plugin.default_params) | set(plugin.search_space) | {"numeric_imputer", "scaler", "threshold", "max_iter"}
            accepted = {name: value for name, value in recommendations.items() if name in allowed_keys and isinstance(value, (str, int, float, bool, type(None)))}
            params.update(accepted)
            reason = advice.get("algorithm_reasons", {}).get(key, plugin.description)
            evidence = list(dict.fromkeys([*prior.get("evidence_ids", []), *advice.get("evidence_ids", [])]))
            preprocessing = advice.get("preprocessing_recommendations", {}).get(key) or plugin.preprocessing
            components["llm_proposal"] = .03 if key in advice.get("candidate_algorithms", []) else 0
            score = sum(components.values())
            components["historical_case_count"] = prior.get("sample_count", 0)
            components["llm_parameters"] = accepted
            plans.append(AlgorithmPlan(algorithm_id, plugin.name, reason, list(preprocessing), params, expected, 0, algorithm_id, "llm_proposed" if accepted else "default", "llm_proposed" if accepted else "default", score, evidence, components))
        return sorted(plans, key=lambda p: -p.search_score)

    @staticmethod
    def _utility(metric: str, value: float, spec: CapabilitySpec) -> float:
        if not METRIC_REGISTRY.is_maximize(metric):
            scale = max(1e-6, abs(spec.metric_thresholds.get(metric, 1)))
            return 1 / (1 + max(0, value) / scale)
        if metric == "r2": return max(0, min(1, (value + 1) / 2))
        return max(0, min(1, value))

