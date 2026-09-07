"""Ground final explanations in candidate measurements and cited retrieval ids."""
from __future__ import annotations

import json
import math
import re
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
        schema = ExplanationContract.model_json_schema()
        schema["properties"]["selected_candidate"]["enum"] = [winner_id]
        schema["properties"]["candidate_comparison"] = {"type": "object", "properties": {key: {"type": "string"} for key in actual}, "required": list(actual), "additionalProperties": False}
        schema["$defs"]["MetricClaim"]["properties"]["candidate_id"]["enum"] = list(actual)
        if allowed_ids:
            schema["properties"]["historical_evidence_used"]["items"]["enum"] = sorted(allowed_ids)
        else:
            schema["properties"]["historical_evidence_used"]["maxItems"] = 0
        payload = {"allowed_current_candidate_ids": list(actual), "requirement": spec.to_dict(), "retrieved_context": knowledge.planning_context, "allowed_historical_evidence_ids": sorted(allowed_ids), "current_candidates": [{"candidate_id": item["plan"]["algorithm_id"], "plan": item["plan"], "validation": {k: item["validation"][k] for k in ("status", "metrics", "errors", "runtime_seconds")}} for item in candidates], "actual_selected_candidate": winner_id, "json_schema": schema}
        for attempt in range(2):
            try:
                raw = self.llm.complete("You are ExplanationAgent. Explain why these plans were tried, which history informed them, current measured comparisons and limitations. Return strict JSON. Never invent measurements or evidence IDs; put numbers only in metric_claims. candidate_comparison describes current_candidates ONLY. For history, only cite run IDs and explain that retrieved cases inform uncertain planning priors. Do not assert historical algorithm performance, reliability, success/failure, or rankings in free text. Describe measured outcomes only for current_candidates. A passing algorithm can still have poor metrics; state small dataset limitations.", json.dumps(payload, ensure_ascii=False), purpose="explanation", generation_config={"json_schema": schema, "max_new_tokens": 2600})
                explanation = ExplanationContract.model_validate(extract_json_object(raw))
                if explanation.selected_candidate != winner_id:
                    raise ValueError("explanation changes actual winner")
                if set(explanation.historical_evidence_used) - allowed_ids:
                    raise ValueError("unknown historical evidence id")
                if set(explanation.candidate_comparison) - set(actual):
                    raise ValueError(f"unknown current candidates: {sorted(set(explanation.candidate_comparison) - set(actual))}; allowed: {list(actual)}")
                for claim in explanation.metric_claims:
                    measured = actual.get(claim.candidate_id, {}).get(claim.metric)
                    if measured is None or not math.isclose(measured, claim.score, rel_tol=1e-5, abs_tol=1e-5):
                        raise ValueError("hallucinated numeric claim")
                validate_comparative_claims(explanation.model_dump(), candidates)
                return {**explanation.model_dump(), "status": "ok", "comparison_check": "passed", "provider": getattr(self.llm, "last_provider", self.provider), "attempts": attempt + 1}
            except Exception as exc:
                payload["validation_error"] = sanitize(str(exc))
        fallback["error"] = payload.get("validation_error")
        return fallback


def validate_comparative_claims(explanation: dict, candidates: list[dict]) -> None:
    """Reject observable metric-order contradictions in supported prose patterns.

    This supplements structured numeric/ID checks; it is not a proof of general
    natural-language entailment. Unchecked narrative still needs human review.
    """
    texts = [explanation.get("why_this_plan", ""), *explanation.get("candidate_comparison", {}).values(), *explanation.get("limitations", [])]
    aliases = {"roc_auc": ["roc-auc", "roc_auc", "auc"], "pr_auc": ["pr-auc", "pr_auc", "average precision"], "f1": ["f1"], "accuracy": ["accuracy"], "r2": ["r2"], "mae": ["mae"], "rmse": ["rmse"]}
    for text in texts:
        for sentence in re.split(r"[.!?](?:\s|$)|[。！？]", text):
            folded = sentence.lower()
            if re.search(r"historical|previous|prior|历史|以往", folded) and re.search(r"\b(higher|lower|better|worse|best|highest|lowest|outperform\w*|underperform\w*)\b|优于|劣于|更高|更低", folded):
                raise ValueError(f"Free-text historical rankings are not supported: {sentence}. Cite run IDs and compare only current candidates.")
            if re.search(r"historical|previous|历史|以往", folded) and re.search(r"\b(passed|failed|succeeded|inconsistent)\b|成功|失败", folded):
                raise ValueError(f"Unstructured historical outcome claims are not supported: {sentence}. Cite cases as uncertain planning priors; describe outcomes for current_candidates only.")
            for left in candidates:
                left_name = left["plan"]["algorithm_name"].lower()
                if left_name not in folded:
                    continue
                for right in candidates:
                    right_name = right["plan"]["algorithm_name"].lower()
                    if left_name == right_name or right_name not in folded:
                        continue
                    direction = re.search(re.escape(left_name) + r".{0,150}?\b(lower|higher|better|worse|outperforms|underperforms)\b.{0,50}?" + re.escape(right_name), folded)
                    if not direction:
                        continue
                    for metric, names in aliases.items():
                        if metric == "roc_auc" and any(name in folded for name in aliases["pr_auc"]):
                            continue
                        if not any(name in folded for name in names):
                            continue
                        a = left["validation"]["metrics"].get(metric)
                        b = right["validation"]["metrics"].get(metric)
                        if a is None or b is None:
                            raise ValueError(f"unmeasured comparison: {left_name}/{right_name} {metric}")
                        word = direction.group(1)
                        greater = word in {"higher", "better", "outperforms"}
                        if word in {"better", "worse", "outperforms", "underperforms"} and metric in {"mae", "rmse"}:
                            greater = not greater
                        if not (a > b if greater else a < b):
                            raise ValueError(f"contradictory comparison: {left_name} {word} {right_name} for {metric}; actual {a} vs {b}. Correct or remove this claim.")
