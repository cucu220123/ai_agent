"""Explainable fusion of document similarity, graph distance and validation context."""
from __future__ import annotations
from typing import Any
from app.experience.retriever import ExperienceRetriever
from app.models import CapabilitySpec


def rerank_evidence(spec: CapabilitySpec, semantic: list[dict[str, Any]], graph: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in semantic:
        source = item["source"]
        key = source.get("id") or source.get("run_id")
        if key:
            by_id[key] = {**item, "semantic_similarity": max(0.0, min(1.0, float(item.get("score", 0)))), "graph_signal": 0.0}
    for node in graph.get("nodes", []):
        if node.get("type") not in {"SourceDocument", "FailureExperience", "RepairExperience", "ValidationRun"}:
            continue
        key = node["id"]
        record = by_id.setdefault(key, {"source": node, "semantic_similarity": 0.0, "backend": "graph"})
        record["graph_signal"] = 1.0 / (1.0 + node.get("graph_distance", 3))
    ranked = []
    for record in by_id.values():
        source = record["source"]
        task = source.get("task_type")
        if task and task != spec.task_type:
            continue
        compatibility = 1.0 if task == spec.task_type else 0.5
        recency = ExperienceRetriever._recency_weight(source.get("timestamp"))
        quality = 1.0 if source.get("status") == "passed" else 0.5
        components = {"semantic": 0.40 * record["semantic_similarity"], "graph": 0.30 * record.get("graph_signal", 0), "task": 0.15 * compatibility, "recency": 0.05 * recency, "validation_quality": 0.10 * quality}
        ranked.append({**record, "score": round(sum(components.values()), 6), "score_components": components})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]

