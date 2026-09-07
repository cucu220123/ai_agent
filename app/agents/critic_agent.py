"""Typed failure analysis with immutable observed evidence and reusable lessons."""
from __future__ import annotations
import json
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from app.llm.contracts import extract_json_object
from app.llm.security import sanitize
from app.models import AlgorithmPlan, CapabilitySpec, ValidationResult


class DiagnosisContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    failure_type: str = Field(min_length=3)
    root_cause: str = Field(min_length=3)
    triggering_condition: str = Field(min_length=3)
    repair_strategy: str = Field(min_length=3)
    reusable_lesson: str = Field(min_length=3)


class CriticAgent:
    def __init__(self, llm=None, provider_name: str = "mock"):
        self.llm, self.provider_name = llm, provider_name

    def run(self, spec: CapabilitySpec, plan: AlgorithmPlan, result: ValidationResult, source_code: str = "", retrieved_experiences: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        experiences = retrieved_experiences or []
        kind = result.failure_type or ("interface_failure" if "missing function" in " ".join(result.errors) else "validation_failure")
        diagnosis = {"failure_type": kind, "root_cause": result.root_cause or (result.errors[0] if result.errors else "none"), "triggering_condition": f"{spec.task_type}: {kind}", "repair_strategy": "Inspect the failed check, restore the protocol and correct preprocessing/parameters; rerun independent validation.", "reusable_lesson": "Current measured validation, including failed attempts, must guide repair.", "provider": "deterministic_fallback", "status": "fallback"}
        if self.llm is not None and self.provider_name != "mock":
            try:
                payload = {"requirement": spec.to_dict(), "plan": plan.to_dict(), "validation": result.to_dict(), "original_code": source_code, "historical_failure_experiences": experiences[:5], "json_schema": DiagnosisContract.model_json_schema()}
                raw = self.llm.complete("You are CriticAgent. Explain the observed failure using only evidence. Return strict JSON. Do not claim a repair succeeded before revalidation.", json.dumps(payload, ensure_ascii=False), purpose="critique", generation_config={"json_schema": DiagnosisContract.model_json_schema()})
                contract = DiagnosisContract.model_validate(extract_json_object(raw))
                diagnosis.update(contract.model_dump(), provider=getattr(self.llm, "last_provider", self.provider_name), status="ok")
            except Exception as exc:
                diagnosis["llm_error"] = sanitize(str(exc))
        diagnosis.update(observed_failure_type=kind, observed_error="\n".join(result.errors)[-4000:], retrieved_experience_ids=[x["id"] for x in experiences if x.get("id")])
        return sanitize(diagnosis)

