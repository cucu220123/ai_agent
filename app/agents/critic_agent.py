from __future__ import annotations

import json
from typing import Any

from app.llm.contracts import extract_json_object
from app.models import AlgorithmPlan, CapabilitySpec, ValidationResult


class CriticAgent:
    """Turn raw validation failures into structured, reusable diagnostics."""

    def __init__(self, llm=None, provider_name: str = "mock"):
        self.llm = llm
        self.provider_name = provider_name

    def run(self, spec: CapabilitySpec, plan: AlgorithmPlan, result: ValidationResult, source_code: str = "") -> dict[str, Any]:
        failure_type = self._classify(result)
        diagnosis = {
            "failure_type": failure_type,
            "root_cause": result.errors[0] if result.errors else "none",
            "triggering_condition": self._trigger(spec, result),
            "observed_error": "\n".join(result.errors + [result.stderr])[-4000:],
            "repair_strategy": self._strategy(failure_type),
            "reusable_lesson": self._lesson(failure_type),
            "provider": self.provider_name,
        }
        if self.llm is not None and self.provider_name != "mock" and result.status != "passed":
            try:
                raw = self.llm.complete(
                    "你是 CriticAgent。只输出 JSON：failure_type, root_cause, triggering_condition, repair_strategy, reusable_lesson。",
                    json.dumps({"spec": spec.to_dict(), "plan": plan.to_dict(), "validation": result.to_dict(), "source_code": source_code[-12000:]}, ensure_ascii=False),
                )
                parsed = extract_json_object(raw)
                if parsed:
                    diagnosis.update({k: parsed[k] for k in diagnosis if k in parsed and parsed[k]})
                    diagnosis["provider"] = self.provider_name
            except Exception as exc:
                diagnosis["llm_error"] = f"{type(exc).__name__}: {exc}"
        return diagnosis

    @staticmethod
    def _classify(result: ValidationResult) -> str:
        text = (" ".join(result.errors) + " " + result.stderr).lower()
        if "syntax" in text or "indent" in text:
            return "syntax_failure"
        if "missing" in text or "interface" in text or "output" in text:
            return "interface_failure"
        if "timeout" in text or "runtime" in text:
            return "runtime_failure"
        if "<" in text or "metric" in text:
            return "metric_underperformance"
        return "validation_failure"

    @staticmethod
    def _trigger(spec: CapabilitySpec, result: ValidationResult) -> str:
        if spec.class_imbalance:
            return "class imbalance or threshold-sensitive evaluation"
        if result.checks.get("output_contract", {}).get("passed") is False:
            return "generated output contract mismatch"
        return "validation feedback"

    @staticmethod
    def _strategy(failure_type: str) -> str:
        return {"syntax_failure": "repair syntax and rerun AST checks", "interface_failure": "restore strict algorithm protocol", "runtime_failure": "inspect traceback and harden preprocessing", "metric_underperformance": "try threshold/config/preprocessing variant", "validation_failure": "inspect failed check and rerun"}.get(failure_type, "inspect failed check")

    @staticmethod
    def _lesson(failure_type: str) -> str:
        return {"syntax_failure": "LLM code must pass AST compilation before execution", "interface_failure": "all algorithms must expose the versioned protocol", "runtime_failure": "test missing values and unseen categories", "metric_underperformance": "historical priors are not a substitute for current validation", "validation_failure": "persist the exact check and evidence"}.get(failure_type, "retain validation evidence")

