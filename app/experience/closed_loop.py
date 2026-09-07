from __future__ import annotations

from typing import Any

from app.knowledge.retriever import RetrieverAgent
from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec


class ClosedLoopAnalyzer:
    """Capture before/after evidence that a written-back run is used by a later task."""

    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.retriever = RetrieverAgent(store)

    def before_after(self, spec: CapabilitySpec) -> dict[str, Any]:
        before = self.retriever.run(spec)
        return {"before": {"historical_cases": before.historical_cases, "graph_trace": before.retrieval_trace, "candidate_algorithms": [a.get("id") for a in before.algorithms]}}

    def after(self, spec: CapabilitySpec, run_id: str) -> dict[str, Any]:
        after = self.retriever.run(spec)
        matching = [case for case in after.historical_cases if case.get("run_id") == run_id]
        return {"after": {"historical_cases": after.historical_cases, "matched_run_ids": [x.get("run_id") for x in matching], "graph_trace": after.retrieval_trace, "candidate_algorithms": [a.get("id") for a in after.algorithms]}}

    def evidence(self, before: dict[str, Any], after: dict[str, Any], run_id: str) -> dict[str, Any]:
        before_ids = [x.get("run_id") for x in before.get("before", {}).get("historical_cases", [])]
        after_ids = [x.get("run_id") for x in after.get("after", {}).get("historical_cases", [])]
        return {"target_run_id": run_id, "before_contains_target": run_id in before_ids, "after_contains_target": run_id in after_ids, "before_case_count": len(before_ids), "after_case_count": len(after_ids)}

    def compare_planning(self, spec: CapabilitySpec, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        before_ids = before.get("before", {}).get("candidate_algorithms", [])
        after_ids = after.get("after", {}).get("candidate_algorithms", [])
        return {"before_candidates": before_ids, "after_candidates": after_ids, "changed": before_ids != after_ids, "matched_run_ids": after.get("after", {}).get("matched_run_ids", [])}
