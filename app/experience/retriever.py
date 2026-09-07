from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from app.models import CapabilitySpec
from app.metrics.registry import METRIC_REGISTRY


class ExperienceRetriever:
    """Retrieve similar validation cases and compute exploitation priors plus exploration bonus."""

    def retrieve(self, spec: CapabilitySpec, runs: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
        results = []
        for run in runs:
            similarity = self._similarity(spec, run)
            if similarity <= 0:
                continue
            results.append({"similarity": round(similarity, 6), **run})
        results.sort(key=lambda x: (x["similarity"], x.get("timestamp", "")), reverse=True)
        return results[:limit]

    def algorithm_priors(self, spec: CapabilitySpec, runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            algo = run.get("algorithm_id")
            if algo:
                grouped.setdefault(algo, []).append(run)
        priors: dict[str, dict[str, float]] = {}
        for algo, items in grouped.items():
            successes = sum(1 for item in items if item.get("status") == "passed")
            primary = METRIC_REGISTRY.primary(spec.task_type, spec.metrics, bool(spec.target_column))
            weighted_values = [(float(item.get("metrics", {}).get(primary)), max(0.05, float(item.get("similarity", 0.0))) * self._recency_weight(item.get("timestamp"))) for item in items if item.get("metrics", {}).get(primary) is not None]
            rates = [float(item.get("runtime_seconds", 0.0)) for item in items]
            weight_sum = sum(weight for _, weight in weighted_values)
            priors[algo] = {
                "success_rate": successes / len(items),
                "historical_score": sum(value * weight for value, weight in weighted_values) / weight_sum if weighted_values else 0.5,
                "primary_metric": primary,
                "maximize": METRIC_REGISTRY.is_maximize(primary),
                "mean_runtime": sum(rates) / len(rates) if rates else 0.0,
                "stability_rate": sum(1 for item in items if item.get("checks", {}).get("stability", {}).get("passed", True)) / len(items),
                "mean_recency_weight": sum(self._recency_weight(item.get("timestamp")) for item in items) / len(items),
                "exploration_bonus": 1.0 / math.sqrt(1.0 + len(items)),
                "sample_count": float(len(items)),
            }
        return priors

    @staticmethod
    def _recency_weight(timestamp: str | None) -> float:
        if not timestamp:
            return 0.7
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            age_days = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 86400)
            return max(0.25, math.exp(-age_days / 180.0))
        except (ValueError, TypeError):
            return 0.7

    def retrieve_failures(self, spec: CapabilitySpec, experiences: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        query_terms = {spec.task_type, spec.domain, spec.target_column, *spec.constraints}
        ranked = []
        for item in experiences:
            text = str(item).lower()
            overlap = sum(1 for term in query_terms if term and str(term).lower() in text)
            repair_bonus = 1 if item.get("repair_history") or item.get("repair_success") else 0
            failure_bonus = 1 if item.get("failure_type") else 0
            ranked.append((overlap + repair_bonus + failure_bonus, item))
        ranked.sort(key=lambda value: value[0], reverse=True)
        return [{"relevance_score": score, **item} for score, item in ranked[:limit] if score > 0]

    @staticmethod
    def _similarity(spec: CapabilitySpec, run: dict[str, Any]) -> float:
        if run.get("task_type") and run.get("task_type") != spec.task_type:
            return 0.0
        score = 0.25
        if run.get("domain") == spec.domain:
            score += 0.25
        if run.get("target") == spec.target_column:
            score += 0.15
        old_features = set(run.get("feature_columns", []))
        new_features = set(spec.feature_columns)
        if old_features and new_features:
            score += 0.30 * len(old_features & new_features) / max(1, len(old_features | new_features))
        if run.get("data_type") == spec.data_type:
            score += 0.10
        old_rows = run.get("dataset_profile", {}).get("rows")
        new_rows = spec.dataset_profile.get("row_count") or spec.dataset_profile.get("rows")
        if old_rows and new_rows:
            score += 0.10 * min(float(old_rows), float(new_rows)) / max(float(old_rows), float(new_rows))
        old_rate = run.get("dataset_profile", {}).get("positive_rate")
        new_rate = spec.class_imbalance.get("positive_rate")
        if old_rate is not None and new_rate is not None:
            score += 0.10 * max(0.0, 1.0 - abs(float(old_rate) - float(new_rate)))
        return min(1.0, score)
