from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agents.planning_context import PlanningContextBuilder
from app.llm.contracts import extract_json_object
from app.models import CapabilitySpec, KnowledgeContext
from app.plugins.registry import DEFAULT_REGISTRY
from app.llm.security import sanitize


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
    hyperparameter_recommendations: dict[str, dict[str, Any]] = Field(default_factory=dict)

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
        # KG IDs carry algorithm_; executable registry IDs do not. These are
        # the same verified entity, so normalize both list and keyed sections.
        def canonical(name):
            if isinstance(name, str) and name.startswith("algorithm_") and name[10:] in DEFAULT_REGISTRY.algorithms:
                return name[10:]
            return name
        if isinstance(value.get("candidate_algorithms"), list):
            value["candidate_algorithms"] = [canonical(name) for name in value["candidate_algorithms"]]
        for section in ("algorithm_reasons", "preprocessing_recommendations", "hyperparameter_recommendations"):
            if isinstance(value.get(section), dict):
                value[section] = {canonical(name): content for name, content in value[section].items()}
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
        compatible = {p.id for p in DEFAULT_REGISTRY.algorithms_for(spec.task_type)}
        allowed = [item.get("id", "").replace("algorithm_", "") for item in knowledge.algorithms if item.get("id", "").replace("algorithm_", "") in compatible and item.get("id") != "algorithm_dummy_classifier"]
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
        # A selected run has a short canonical id; its exact candidate/version
        # tuple is a verified alias, not another experiment. Only admit aliases
        # whose canonical id survives context selection.
        evidence_aliases = {}
        for case in knowledge.historical_cases:
            canonical = case.get("run_id")
            version = re.search(r"_v(\d+)$", str(case.get("version_id", "")))
            if canonical in context_trace["final_evidence_ids"] and version and case.get("workflow_run_id") and case.get("candidate_id"):
                alias = f"{case['workflow_run_id']}:{case['candidate_id']}:v{version.group(1)}"
                evidence_aliases[alias] = canonical
        previous = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                payload = {"planning_context": context, "allowed_algorithms": allowed, "allowed_evidence_ids": context_trace["final_evidence_ids"], "verified_evidence_aliases": evidence_aliases, "json_schema": PlannerAdviceContract.model_json_schema(), "execution_rules": "For each candidate recommend concrete preprocessing and a small hyperparameter dictionary. Keep user metric thresholds unchanged. Justify candidates using supplied evidence ids; preserve exploration. Supported configuration keys include C, max_iter, n_estimators, max_depth, learning_rate, min_samples_leaf, class_weight, scaler, numeric_imputer, threshold, max_features, ngram_max, contamination."}
                if attempt > 1:
                    payload.update({"repair_previous_json": previous[:6000], "validation_error": trace["attempts"][-1].get("error")})
                raw = self.llm.complete("You are PlannerAgent. Return one JSON object matching json_schema. Use double quotes and null/true/false, never Python None/True/False. preprocessing_recommendations and hyperparameter_recommendations must be keyed by candidate algorithm IDs, not feature names. Include concrete parameter dictionaries. All evidence_ids must come from supplied evidence. Do not invent outcomes. Use sklearn-only dependencies, no SMOTE unless explicitly supported.", json.dumps(payload, ensure_ascii=False), purpose="planning", generation_config={"json_schema": PlannerAdviceContract.model_json_schema()}) or ""
                previous = raw
                parsed = PlannerAdviceContract.normalize_candidate(extract_json_object(raw))
                try:
                    json.loads(raw)
                    syntax_correction = False
                except ValueError:
                    syntax_correction = parsed is not None
                advice = PlannerAdviceContract.model_validate(parsed)
                corrections = {value: evidence_aliases[value] for value in advice.evidence_ids if value not in context_trace["final_evidence_ids"] and value in evidence_aliases}
                advice.evidence_ids = list(dict.fromkeys(corrections.get(value, value) for value in advice.evidence_ids))
                unknown_algorithms = set(advice.candidate_algorithms) - set(allowed)
                unknown_evidence = set(advice.evidence_ids) - set(context_trace["final_evidence_ids"])
                if set(advice.preprocessing_recommendations) - set(allowed) - {"global"}:
                    raise ValueError("preprocessing_recommendations keys must be candidate algorithm IDs")
                if unknown_algorithms:
                    raise ValueError(f"unsupported algorithms: {sorted(unknown_algorithms)}")
                if unknown_evidence:
                    raise ValueError(f"hallucinated evidence ids: {sorted(unknown_evidence)}")
                result = advice.model_dump()
                trace.update({"status": "ok", "evidence_id_corrections": corrections, "syntax_correction": syntax_correction, "schema_valid": True, "semantic_valid": True, "attempt_count": attempt, "provider": getattr(self.llm, "last_provider", type(self.llm).__name__), "model": getattr(self.llm, "model", None), "token_usage": getattr(self.llm, "last_usage", {}), "generation": getattr(self.llm, "last_generation", {})})
                trace["attempts"].append({"attempt": attempt, "status": "accepted"})
                return result, sanitize(trace)
            except Exception as exc:
                trace["attempts"].append({"attempt": attempt, "status": "rejected", "error": f"{type(exc).__name__}: {exc}"[:1500], "rejected_output": sanitize(previous[:5000])})
        trace["fallback_reason"] = trace["attempts"][-1]["error"] if trace["attempts"] else "no provider"
        return fallback, sanitize(trace)
