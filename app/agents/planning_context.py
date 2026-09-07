from __future__ import annotations

import json
from typing import Any

from app.models import CapabilitySpec, KnowledgeContext


class PlanningContextBuilder:
    """Compress retrieval output into an evidence-only, budgeted planning context."""

    def __init__(self, max_chars: int = 12000):
        self.max_chars = max_chars

    def build(self, spec: CapabilitySpec, knowledge: KnowledgeContext) -> tuple[dict[str, Any], dict[str, Any]]:
        graph_candidates = knowledge.graph_evidence.get("serialized", {}).get("candidate_algorithms", [])[:5]
        historical = knowledge.historical_cases[:5]
        failures = [item for item in knowledge.experiences if item.get("kind") in {"failure", "candidate_failure", "candidate_repaired", "extracted_failure"}][:5]
        semantic = knowledge.semantic_evidence[:5]
        context = {
            "requirement": {
                "domain": spec.domain, "capability_name": spec.capability_name, "task_type": spec.task_type,
                "data_type": spec.data_type, "target": spec.target_column, "features": spec.feature_columns,
                "metrics": spec.metrics, "thresholds": spec.metric_thresholds, "constraints": spec.constraints,
                "latency_requirement_ms": spec.latency_requirement_ms,
                "interpretability_requirement": spec.interpretability_requirement,
                "resource_constraints": spec.resource_constraints,
                "class_imbalance": spec.class_imbalance,
            },
            "graph_candidates": graph_candidates,
            "similar_historical_runs": historical,
            "failure_and_repair_experiences": failures,
            "source_evidence": semantic,
        }
        serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > self.max_chars:
            context["source_evidence"] = semantic[:2]
            context["failure_and_repair_experiences"] = failures[:3]
            context["similar_historical_runs"] = historical[:3]
            serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > self.max_chars:
            context["graph_candidates"] = graph_candidates[:3]
            serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        trace = {
            "raw_retrieved_items": {
                "graph_nodes": len(knowledge.graph_evidence.get("nodes", [])),
                "graph_candidates": len(knowledge.graph_evidence.get("serialized", {}).get("candidate_algorithms", [])),
                "historical_runs": len(knowledge.historical_cases),
                "experiences": len(knowledge.experiences),
                "semantic_items": len(knowledge.semantic_evidence),
            },
            "items_after_rerank": {"graph_candidates": len(context["graph_candidates"]), "historical_runs": len(context["similar_historical_runs"]), "experiences": len(context["failure_and_repair_experiences"]), "source_evidence": len(context["source_evidence"])},
            "context_budget_chars": self.max_chars,
            "final_prompt_chars": len(serialized),
            "estimated_prompt_tokens": max(1, len(serialized) // 3),
            "final_evidence_ids": self._evidence_ids(context),
        }
        return context, trace

    @staticmethod
    def _evidence_ids(context: dict[str, Any]) -> list[str]:
        ids = []
        for item in context.get("graph_candidates", []):
            ids.extend(item.get("evidence_node_ids", []))
            if item.get("algorithm_id"):
                ids.append(item["algorithm_id"])
        for item in context.get("similar_historical_runs", []):
            if item.get("run_id"):
                ids.append(item["run_id"])
        return list(dict.fromkeys(ids))

