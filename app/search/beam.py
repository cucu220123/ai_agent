from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.models import AlgorithmPlan, CapabilitySpec
from app.plugins.registry import DEFAULT_REGISTRY


@dataclass
class SearchTrace:
    strategy: str
    beam_width: int
    expanded: int
    selected: list[str]
    scores: dict[str, float]
    expansions: list[dict[str, Any]] = field(default_factory=list)
    pruned: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "beam_width": self.beam_width, "expanded": self.expanded, "selected": self.selected, "scores": self.scores, "expansions": self.expansions, "pruned": self.pruned}


class BeamSearchPlanner:
    """Search real algorithm/preprocessing/hyperparameter states and retain diverse finalists."""

    def search(self, plans: list[AlgorithmPlan], spec: CapabilitySpec, beam_width: int = 3) -> tuple[list[AlgorithmPlan], SearchTrace]:
        states: list[AlgorithmPlan] = []
        expansions: list[dict[str, Any]] = []
        for parent in plans:
            plugin = DEFAULT_REGISTRY.algorithms.get((parent.base_algorithm_id or parent.algorithm_id).replace("algorithm_", ""))
            for action in self._actions(parent, spec):
                params = {**parent.hyperparameters, **action["parameters"]}
                state_id = parent.algorithm_id + "__" + self._slug(action["name"])
                state = AlgorithmPlan(
                    algorithm_id=state_id, algorithm_name=parent.algorithm_name, rationale=parent.rationale,
                    preprocessing=action["preprocessing"], hyperparameters=params,
                    expected_metrics=parent.expected_metrics, priority=parent.priority,
                    base_algorithm_id=parent.base_algorithm_id or parent.algorithm_id,
                    preprocessing_variant=action["preprocessing_variant"], config_variant=action["name"],
                    search_score=0.0, evidence_ids=parent.evidence_ids,
                )
                state.search_score = self._score(parent, state, spec)
                states.append(state)
                expansions.append({"parent_state": parent.algorithm_id, "expansion_action": action, "child_state": state_id, "score": round(state.search_score, 6)})
        ranked = sorted(states, key=lambda item: item.search_score, reverse=True)
        selected: list[AlgorithmPlan] = []
        seen_bases: set[str] = set()
        for candidate in ranked:
            base = candidate.base_algorithm_id or candidate.algorithm_id
            if base not in seen_bases:
                selected.append(candidate)
                seen_bases.add(base)
            if len(selected) >= max(1, beam_width):
                break
        if len(selected) < max(1, beam_width):
            for candidate in ranked:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= max(1, beam_width):
                    break
        selected_ids = {item.algorithm_id for item in selected}
        pruned = [{"state": item.algorithm_id, "score": round(item.search_score, 6), "prune_reason": "outside_top_diverse_beam"} for item in ranked if item.algorithm_id not in selected_ids]
        scores = {item.algorithm_id: round(item.search_score, 6) for item in states}
        return selected, SearchTrace("combinatorial_beam_search", max(1, beam_width), len(states), [item.algorithm_id for item in selected], scores, expansions, pruned)

    def _actions(self, plan: AlgorithmPlan, spec: CapabilitySpec) -> list[dict[str, Any]]:
        key = (plan.base_algorithm_id or plan.algorithm_id).replace("algorithm_", "")
        plugin = DEFAULT_REGISTRY.algorithms.get(key)
        if plugin and plugin.search_space and key not in {"logistic_regression", "random_forest", "gradient_boosting", "random_forest_regressor", "isolation_forest", "tfidf_logistic_regression"}:
            actions = [{"name": f"{key}_{index}", "parameters": {key: value}, "preprocessing_variant": ";".join(plugin.preprocessing), "preprocessing": plugin.preprocessing} for index, (key, value) in enumerate((item, value) for item, values in plugin.search_space.items() for value in values)]
            return actions or [self._action("default", {})]
        imbalance = bool(spec.class_imbalance.get("is_imbalanced") or spec.class_imbalance.get("positive_rate", 1.0) < 0.25)
        if key == "logistic_regression":
            return [
                self._action("standard_c1", {"C": 1.0, "class_weight": None, "scaler": "standard", "threshold": 0.5, "numeric_imputer": "median"}),
                self._action("balanced_c1", {"C": 1.0, "class_weight": "balanced", "scaler": "standard", "threshold": 0.4, "numeric_imputer": "median"}),
                self._action("regularized_robust", {"C": 0.3, "class_weight": "balanced" if imbalance else None, "scaler": "robust", "threshold": 0.45, "numeric_imputer": "median"}),
                self._action("weak_regularization", {"C": 3.0, "class_weight": None, "scaler": "standard", "threshold": 0.5, "numeric_imputer": "mean"}),
            ]
        if key == "random_forest":
            return [
                self._action("rf_fast", {"n_estimators": 80, "max_depth": 6, "min_samples_leaf": 2, "class_weight": "balanced"}),
                self._action("rf_balanced", {"n_estimators": 180, "max_depth": 10, "min_samples_leaf": 2, "class_weight": "balanced"}),
                self._action("rf_deep", {"n_estimators": 240, "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced_subsample"}),
                self._action("rf_regularized", {"n_estimators": 160, "max_depth": 7, "min_samples_leaf": 5, "class_weight": "balanced"}),
            ]
        if key == "gradient_boosting":
            return [
                self._action("gb_fast", {"n_estimators": 80, "learning_rate": 0.08, "max_depth": 2}),
                self._action("gb_baseline", {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3}),
                self._action("gb_slow_deep", {"n_estimators": 180, "learning_rate": 0.03, "max_depth": 3}),
                self._action("gb_shallow", {"n_estimators": 150, "learning_rate": 0.04, "max_depth": 2}),
            ]
        if key == "random_forest_regressor":
            return [self._action("reg_fast", {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 2}), self._action("reg_accurate", {"n_estimators": 220, "max_depth": 12, "min_samples_leaf": 1})]
        if key == "tfidf_logistic_regression":
            return [self._action("tfidf_unigram", {"C": 1.0, "max_features": 3000, "ngram_max": 1}), self._action("tfidf_bigram", {"C": 1.0, "max_features": 5000, "ngram_max": 2}), self._action("tfidf_regularized", {"C": 0.4, "max_features": 8000, "ngram_max": 2})]
        if key == "isolation_forest":
            return [self._action("isolation_auto", {"n_estimators": 160, "contamination": "auto"}), self._action("isolation_05", {"n_estimators": 200, "contamination": 0.05}), self._action("isolation_10", {"n_estimators": 120, "contamination": 0.1})]
        return [self._action("default", {})]

    @staticmethod
    def _action(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        preprocessing_variant = f"imputer={parameters.get('numeric_imputer', 'median')};scaler={parameters.get('scaler', 'none')}"
        preprocessing = [step for step in [parameters.get("numeric_imputer", "median") + " imputation", parameters.get("scaler")] if step and step != "none"]
        return {"name": name, "parameters": parameters, "preprocessing_variant": preprocessing_variant, "preprocessing": preprocessing}

    @staticmethod
    def _score(parent: AlgorithmPlan, state: AlgorithmPlan, spec: CapabilitySpec) -> float:
        score = float(parent.search_score or 0.5)
        params = state.hyperparameters
        if (spec.interpretability_requirement == "high" or "prefer_interpretable" in spec.constraints) and state.algorithm_name == "Logistic Regression":
            score += 0.12
        if (spec.latency_requirement_ms or "low_latency" in spec.constraints) and (params.get("n_estimators", 0) or 0) > 180:
            score -= 0.08
        if params.get("class_weight") and spec.task_type == "binary_classification":
            score += 0.025
        if params.get("scaler") == "robust":
            score += 0.01
        if params.get("n_estimators", 0) and params.get("n_estimators", 0) <= 80:
            score += 0.015
        return score

    @staticmethod
    def _slug(value: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
