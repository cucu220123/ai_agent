"""Ground final explanations in candidate measurements and cited retrieval ids."""
from __future__ import annotations

import json
import math
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from app.llm.contracts import extract_json_object
from app.llm.security import sanitize
from app.models import CapabilitySpec, KnowledgeContext


class MetricClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    metric: str
    score: float


class ExplanationContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_candidate: str
    why_this_plan: str = Field(min_length=10)
    historical_evidence_used: list[str]
    candidate_comparison: dict[str, str]
    metric_claims: list[MetricClaim]
    limitations: list[str]


class ExplanationAgent:
    def __init__(self, llm=None, provider: str = "mock"):
        self.llm, self.provider = llm, provider

    def run(self, spec: CapabilitySpec, knowledge: KnowledgeContext, candidates: list[dict], winner: dict) -> dict[str, Any]:
        winner_id = winner["plan"]["algorithm_id"]
        actual = {item["plan"]["algorithm_id"]: item["validation"]["metrics"] for item in candidates}
        allowed_ids = set(knowledge.planner_advice.get("evidence_ids", [])) | {x["run_id"] for x in knowledge.historical_cases}
        fallback = {"status": "deterministic_fallback", "selected_candidate": winner_id, "why_this_plan": "Candidates were filtered by task/constraints, prioritized with contextual history and exploration, and ranked using current independently recomputed validation metrics.", "historical_evidence_used": sorted(allowed_ids), "candidate_comparison": {item["plan"]["algorithm_id"]: item["validation"]["status"] for item in candidates}, "metric_claims": [{"candidate_id": key, "metric": metric, "score": score} for key, values in actual.items() for metric, score in values.items()], "limitations": ["Synthetic demo data; holdout metrics are not production performance.", "Prototype process/audit sandbox is not a production hostile-code security boundary.", "Generated free-text explanations require human review; numeric claims and evidence ids are checked."]}
        if self.provider == "mock" or self.llm is None:
            return fallback
        payload = {"requirement": spec.to_dict(), "retrieved_context": knowledge.planning_context, "allowed_historical_evidence_ids": sorted(allowed_ids), "current_candidates": [{"candidate_id": item["plan"]["algorithm_id"], "plan": item["plan"], "validation": {k: item["validation"][k] for k in ("status", "metrics", "errors", "runtime_seconds")}} for item in candidates], "actual_selected_candidate": winner_id, "json_schema": ExplanationContract.model_json_schema()}
        for attempt in range(2):
            try:
                raw = self.llm.complete("You are ExplanationAgent. Explain why these plans were tried, which history informed them, current measured comparisons and limitations. Return strict JSON. Never invent measurements or evidence IDs; put numbers only in metric_claims.", json.dumps(payload, ensure_ascii=False), purpose="explanation", generation_config={"json_schema": ExplanationContract.model_json_schema(), "max_new_tokens": 2600})
                explanation = ExplanationContract.model_validate(extract_json_object(raw))
                if explanation.selected_candidate != winner_id:
                    raise ValueError("explanation changes actual winner")
                if set(explanation.historical_evidence_used) - allowed_ids:
                    raise ValueError("unknown historical evidence id")
                if set(explanation.candidate_comparison) - set(actual):
                    raise ValueError("unknown candidate")
                for claim in explanation.metric_claims:
                    measured = actual.get(claim.candidate_id, {}).get(claim.metric)
                    if measured is None or not math.isclose(measured, claim.score, rel_tol=1e-5, abs_tol=1e-5):
                        raise ValueError("hallucinated numeric claim")
                return {**explanation.model_dump(), "status": "ok", "provider": getattr(self.llm, "last_provider", self.provider), "attempts": attempt + 1}
            except Exception as exc:
                payload["validation_error"] = sanitize(str(exc))
        fallback["error"] = payload.get("validation_error")
        return fallback
