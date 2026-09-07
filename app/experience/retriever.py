from __future__ import annotations

import math
from typing import Any

from app.models import CapabilitySpec


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
            weights = [max(0.05, float(item.get("similarity", 0.0))) for item in items]
            successes = sum(1 for item in items if item.get("status") == "passed")
            aucs = [float(item.get("metrics", {}).get("roc_auc", 0.0)) for item in items if item.get("metrics", {}).get("roc_auc") is not None]
            rates = [float(item.get("runtime_seconds", 0.0)) for item in items]
            weight_sum = sum(weights)
            priors[algo] = {
                "success_rate": successes / len(items),
                "historical_score": sum(a * w for a, w in zip(aucs, weights)) / weight_sum if aucs else 0.5,
                "mean_runtime": sum(rates) / len(rates) if rates else 0.0,
                "exploration_bonus": 1.0 / math.sqrt(1.0 + len(items)),
                "sample_count": float(len(items)),
            }
        return priors

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
        return min(1.0, score)

