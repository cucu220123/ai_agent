from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agents.planning_context import PlanningContextBuilder
from app.llm.contracts import extract_json_object
from app.models import CapabilitySpec, KnowledgeContext


class PlannerAdviceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_algorithms: list[str] = Field(min_length=1)
    preprocessing_recommendations: dict[str, list[str]]
    metric_strategy: dict[str, Any]
    constraint_analysis: list[str]
    evidence_ids: list[str] = Field(min_length=1)
    algorithm_reasons: dict[str, str]
    risks: list[str]
    confidence: float = Field(ge=0, le=1)

    @field_validator("candidate_algorithms")
    @classmethod
    def unique_algorithms(cls, value: list[str]) -> list[str]:
        result = list(dict.fromkeys(x.strip() for x in value if x.strip()))
        if not result:
            raise ValueError("candidate_algorithms cannot be empty")
        return result

    @model_validator(mode="after")
    def reasons_cover_candidates(self):
        missing = set(self.candidate_algorithms) - set(self.algorithm_reasons)
        if missing:
            raise ValueError(f"algorithm_reasons missing {sorted(missing)}")
        if self.confidence < 0.4:
            raise ValueError("confidence below semantic acceptance threshold")
        return self

    @classmethod
    def normalize_candidate(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if isinstance(value.get("preprocessing_recommendations"), list):
            value["preprocessing_recommendations"] = {str(item.get("algorithm", "global")) if isinstance(item, dict) else "global": (item.get("steps", []) if isinstance(item, dict) else [str(item)]) for item in value["preprocessing_recommendations"]}
        if isinstance(value.get("constraint_analysis"), dict):
            value["constraint_analysis"] = [f"{key}={item}" for key, item in value["constraint_analysis"].items()]
        if isinstance(value.get("confidence"), bool) or value.get("confidence") is None:
            value["confidence"] = 0.0
        return value


class AdvisorAgent:
    """Strict Planner advice with compressed evidence, retry, and explicit fallback."""

    def __init__(self, llm: Any, max_context_chars: int = 12000, max_attempts: int = 2):
        self.llm = llm
        self.context_builder = PlanningContextBuilder(max_context_chars)
        self.max_attempts = max_attempts

    def run(self, spec: CapabilitySpec, knowledge: KnowledgeContext) -> tuple[dict[str, Any], dict[str, Any]]:
        context, context_trace = self.context_builder.build(spec, knowledge)
        allowed = [item.get("id", "").replace("algorithm_", "") for item in knowledge.algorithms]
        fallback = {
            "candidate_algorithms": [x for x in spec.candidate_algorithms if x in allowed] or allowed,
            "preprocessing_recommendations": {x: [] for x in allowed},
            "metric_strategy": {"primary": spec.metrics[0] if spec.metrics else None, "thresholds": spec.metric_thresholds},
            "constraint_analysis": spec.constraints,
            "evidence_ids": context_trace["final_evidence_ids"] or ["cold_start"],
            "algorithm_reasons": {x: "deterministic compatibility fallback" for x in allowed},
            "risks": ["LLM planner advice unavailable or rejected"],
            "confidence": 0.0,
        }
        trace: dict[str, Any] = {"status": "fallback", "schema_valid": False, "semantic_valid": False, "context": context_trace, "attempts": [], "raw_retrieved_items": context_trace["raw_retrieved_items"], "items_after_rerank": context_trace["items_after_rerank"], "final_prompt_evidence": context, "prompt_token_count": context_trace["estimated_prompt_tokens"]}
        if self.llm is None:
            return fallback, trace
        previous = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                payload = {"planning_context": context, "allowed_algorithms": allowed, "required_fields": list(PlannerAdviceContract.model_fields)}
                if attempt > 1:
                    payload.update({"repair_previous_json": previous[:6000], "validation_error": trace["attempts"][-1].get("error")})
                raw = self.llm.complete("你是 PlannerAgent。只返回严格 JSON，不要预测结果、Markdown 或思考过程。所有 evidence_ids 必须来自上下文。", json.dumps(payload, ensure_ascii=False), purpose="planning") or ""
                previous = raw
                parsed = PlannerAdviceContract.normalize_candidate(extract_json_object(raw))
                advice = PlannerAdviceContract.model_validate(parsed)
                unknown_algorithms = set(advice.candidate_algorithms) - set(allowed)
                unknown_evidence = set(advice.evidence_ids) - set(context_trace["final_evidence_ids"])
                if unknown_algorithms:
                    raise ValueError(f"unsupported algorithms: {sorted(unknown_algorithms)}")
                if unknown_evidence:
                    raise ValueError(f"hallucinated evidence ids: {sorted(unknown_evidence)}")
                result = advice.model_dump()
                trace.update({"status": "ok", "schema_valid": True, "semantic_valid": True, "attempt_count": attempt, "provider": getattr(self.llm, "last_provider", type(self.llm).__name__), "model": getattr(self.llm, "model", None), "token_usage": getattr(self.llm, "last_usage", {}), "generation": getattr(self.llm, "last_generation", {})})
                trace["attempts"].append({"attempt": attempt, "status": "accepted"})
                return result, trace
            except Exception as exc:
                trace["attempts"].append({"attempt": attempt, "status": "rejected", "error": f"{type(exc).__name__}: {exc}"[:1500]})
        trace["fallback_reason"] = trace["attempts"][-1]["error"] if trace["attempts"] else "no provider"
        return fallback, trace
