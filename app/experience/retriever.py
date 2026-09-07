"""Context-sensitive case retrieval. Measurements are priors, never final decisions."""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Any
from app.models import CapabilitySpec
from app.metrics.registry import METRIC_REGISTRY


class ExperienceRetriever:
    def retrieve(self, spec: CapabilitySpec, runs: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
        results = []
        for run in runs:
            if run.get("origin") == "llm_extracted":
                continue  # Textual claims are evidence, not trusted measurements.
            similarity = self._similarity(spec, run)
            if similarity > 0:
                results.append({**run, "similarity": round(similarity, 6)})
        return sorted(results, key=lambda x: (x["similarity"], x.get("timestamp", "")), reverse=True)[:limit]

    def algorithm_priors(self, spec: CapabilitySpec, runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            algo = run.get("base_algorithm_id") or run.get("algorithm_id")
            if algo and self._similarity(spec, run) > 0 and run.get("origin") != "llm_extracted":
                grouped.setdefault(algo.split("__", 1)[0], []).append(run)
        priors = {}
        primary = METRIC_REGISTRY.primary(spec.task_type, spec.metrics, bool(spec.target_column))
        for algorithm, items in grouped.items():
            weights = [max(0.01, float(item.get("similarity", self._similarity(spec, item)))) * self._recency_weight(item.get("timestamp")) for item in items]
            total = sum(weights)
            measured = [(float(item["metrics"][primary]), weight) for item, weight in zip(items, weights) if primary in item.get("metrics", {}) and math.isfinite(float(item["metrics"][primary]))]
            priors[algorithm] = {
                "success_rate": sum(w for item, w in zip(items, weights) if item.get("status") == "passed") / total,
                "historical_score": sum(v * w for v, w in measured) / sum(w for _, w in measured) if measured else None,
                "primary_metric": primary, "maximize": METRIC_REGISTRY.is_maximize(primary),
                "mean_runtime": sum(float(item.get("runtime_seconds", 0)) * w for item, w in zip(items, weights)) / total,
                "mean_memory_mb": sum(float(item.get("resource_usage", {}).get("max_rss_kb") or 0) / 1024 * w for item, w in zip(items, weights)) / total,
                "stability_rate": sum(w for item, w in zip(items, weights) if item.get("checks", {}).get("stability", {}).get("passed", False)) / total,
                "exploration_bonus": 1 / math.sqrt(1 + len(items)),
                "sample_count": len(items), "effective_weight": total,
                "evidence_ids": [item["run_id"] for item in items],
            }
        return priors

    @staticmethod
    def _recency_weight(timestamp: str | None) -> float:
        if not timestamp: return 0.7
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=timezone.utc)
            age = max(0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86400)
            return max(0.1, math.exp(-age / 180))
        except (ValueError, TypeError):
            return 0.7

    def retrieve_failures(self, spec: CapabilitySpec, experiences: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        results = []
        for item in experiences:
            if item.get("task_type") and item["task_type"] != spec.task_type:
                continue
            score = self._similarity(spec, item)
            if not score: continue
            score += 0.1 * bool(item.get("repair_success")) + 0.05 * bool(item.get("reusable_lesson"))
            results.append({**item, "relevance_score": score})
        return sorted(results, key=lambda x: (x["relevance_score"], x.get("timestamp", "")), reverse=True)[:limit]

    @staticmethod
    def _similarity(spec: CapabilitySpec, run: dict[str, Any]) -> float:
        if run.get("task_type") and run["task_type"] != spec.task_type: return 0.0
        if run.get("data_type") and run["data_type"] != spec.data_type: return 0.0
        old = run.get("dataset_profile", {})
        new = spec.dataset_profile
        features_old = set(run.get("feature_columns") or old.get("feature_columns", []))
        features_new = set(spec.feature_columns)
        feature_match = len(features_old & features_new) / max(1, len(features_old | features_new)) if features_old and features_new else 0.5
        rows_old, rows_new = old.get("rows"), new.get("rows", new.get("row_count"))
        size_match = min(rows_old, rows_new) / max(rows_old, rows_new) if rows_old and rows_new else 0.5
        numeric_match = 1 - abs(old.get("numeric_fraction", 0.5) - new.get("numeric_fraction", 0.5))
        old_rate, new_rate = old.get("positive_rate"), spec.class_imbalance.get("positive_rate")
        balance_match = 1 - abs(old_rate - new_rate) if old_rate is not None and new_rate is not None else 0.5
        resource_match = 1.0
        memory_limit = spec.resource_constraints.get("max_memory_mb")
        previous_memory = (run.get("resource_usage", {}).get("max_rss_kb") or 0) / 1024
        if memory_limit and previous_memory:
            resource_match *= min(1, float(memory_limit) / previous_memory)
        score = .25 + .15 * (run.get("domain") == spec.domain) + .10 * (run.get("target") == spec.target_column) + .15 * feature_match + .10 * size_match + .10 * numeric_match + .10 * balance_match + .05 * resource_match
        return min(1, score)

