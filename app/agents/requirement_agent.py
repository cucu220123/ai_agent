from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.agents.parser_agent import ParserAgent
from app.llm.contracts import extract_json_object
from app.models import CapabilitySpec


TaskType = Literal["binary_classification", "multiclass_classification", "regression", "anomaly_detection", "text_classification"]
DataType = Literal["tabular", "text", "image", "timeseries"]

METRIC_ALIASES = {
    "auc": "roc_auc", "roc-auc": "roc_auc", "roc_auc": "roc_auc", "pr-auc": "pr_auc", "pr_auc": "pr_auc",
    "f1-score": "f1", "f1_score": "f1", "f1": "f1", "accuracy": "accuracy", "precision": "precision",
    "recall": "recall", "mae": "mae", "rmse": "rmse", "r2": "r2", "r²": "r2",
}


class RequirementContract(BaseModel):
    """Strict LLM contract: required semantic fields cannot be filled by defaults."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    domain: str = Field(min_length=2)
    capability_name: str = Field(min_length=2)
    task_type: TaskType
    data_type: DataType
    target: str
    input_schema: dict[str, str] = Field(default_factory=dict)
    output_schema: dict[str, str] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    metrics: list[str] = Field(min_length=1)
    thresholds: dict[str, float] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    latency_requirement_ms: float | None = Field(default=None, ge=0)
    interpretability_requirement: Literal["none", "low", "medium", "high"] | None = None
    resource_constraint: dict[str, Any] = Field(default_factory=dict)
    probability_output_required: bool
    class_imbalance: dict[str, Any] = Field(default_factory=dict)
    candidate_hints: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @field_validator("domain")
    @classmethod
    def domain_must_be_specific(cls, value: str) -> str:
        if value.lower() in {"general", "generic", "unknown", "算法", "一般"}:
            raise ValueError("domain must be specific")
        return value

    @field_validator("capability_name")
    @classmethod
    def capability_must_be_specific(cls, value: str) -> str:
        if value.lower() in {"算法能力", "algorithm capability", "unknown", "task"}:
            raise ValueError("capability_name must be specific")
        return value

    @field_validator("metrics")
    @classmethod
    def normalize_metrics(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            key = value.strip().lower().replace(" ", "_")
            metric = METRIC_ALIASES.get(key, key)
            if metric and metric not in normalized:
                normalized.append(metric)
        if not normalized:
            raise ValueError("at least one meaningful metric is required")
        return normalized

    @field_validator("thresholds")
    @classmethod
    def normalize_thresholds(cls, values: dict[str, float]) -> dict[str, float]:
        return {METRIC_ALIASES.get(str(key).lower().replace(" ", "_"), str(key).lower()): float(value) for key, value in values.items()}

    @model_validator(mode="after")
    def validate_task_semantics(self):
        if self.task_type != "anomaly_detection" and not self.target:
            raise ValueError("supervised task requires target")
        if self.task_type == "anomaly_detection" and self.target and self.target not in self.input_schema:
            # A supervised anomaly label is allowed only if explicitly represented.
            raise ValueError("anomaly target must be empty or present in schema")
        if self.probability_output_required and self.task_type not in {"binary_classification", "multiclass_classification", "text_classification"}:
            raise ValueError("probability output is incompatible with task type")
        return self


class RequirementUnderstandingAgent:
    """LLM-first requirement understanding with strict retry and deterministic fallback."""

    def __init__(self, llm=None, provider_name: str = "mock", max_attempts: int = 2):
        self.llm = llm
        self.provider_name = provider_name
        self.max_attempts = max_attempts
        self.fallback = ParserAgent()

    def run(self, description: str, dataset_path: str | Path | None = None) -> tuple[CapabilitySpec, dict[str, Any]]:
        fallback = self.fallback.run(description, str(dataset_path) if dataset_path else None)
        trace: dict[str, Any] = {"provider": self.provider_name, "status": "fallback", "schema_valid": False, "semantic_valid": False, "attempts": []}
        if self.llm is None or self.provider_name == "mock":
            trace["fallback_reason"] = "mock_or_missing_provider"
            return fallback, trace

        prompt = self._prompt(description, dataset_path)
        previous = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                if attempt > 1:
                    prompt = json.dumps({"instruction": "Repair the previous JSON. Return only a complete object matching the schema.", "previous_output": previous[:8000], "validation_error": trace["attempts"][-1].get("error"), "original_request": json.loads(self._prompt(description, dataset_path))}, ensure_ascii=False)
                raw = self.llm.complete("你是 RequirementUnderstandingAgent。只返回严格 JSON，不要 Markdown 或思考过程。", prompt, purpose="requirement") or ""
                previous = raw
                parsed = extract_json_object(raw)
                contract = RequirementContract.model_validate(parsed)
                quality_errors = self._quality_gate(contract, description, dataset_path)
                if quality_errors:
                    raise ValueError("; ".join(quality_errors))
                spec, corrections = self._to_spec(contract, fallback, description, dataset_path)
                trace.update({"provider": getattr(self.llm, "last_provider", self.provider_name), "model": getattr(self.llm, "model", None), "status": "ok", "schema_valid": True, "semantic_valid": True, "semantic_corrections": corrections, "attempt_count": attempt, "response_preview": raw[:1000], "token_usage": getattr(self.llm, "last_usage", {}), "generation": getattr(self.llm, "last_generation", {})})
                trace["attempts"].append({"attempt": attempt, "status": "accepted"})
                return spec, trace
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                trace["attempts"].append({"attempt": attempt, "status": "rejected", "error": f"{type(exc).__name__}: {exc}"[:1500]})
            except Exception as exc:
                trace["attempts"].append({"attempt": attempt, "status": "provider_error", "error": f"{type(exc).__name__}: {exc}"[:1500]})
                break
        trace["fallback_reason"] = trace["attempts"][-1].get("error", "invalid structured output") if trace["attempts"] else "provider unavailable"
        return fallback, trace

    def _quality_gate(self, contract: RequirementContract, description: str, dataset_path: str | Path | None) -> list[str]:
        errors = []
        if contract.confidence < 0.45:
            errors.append("confidence below 0.45")
        lowered = description.lower()
        required_metrics = []
        for alias, canonical in (("roc-auc", "roc_auc"), ("auc", "roc_auc"), ("f1", "f1"), ("mae", "mae"), ("rmse", "rmse"), ("r2", "r2")):
            if alias in lowered and canonical not in required_metrics:
                required_metrics.append(canonical)
        missing = set(required_metrics) - set(contract.metrics)
        if missing:
            errors.append(f"metrics omit user requirements: {sorted(missing)}")
        if dataset_path:
            try:
                columns = pd.read_csv(dataset_path, nrows=0).columns.tolist()
                if contract.task_type != "anomaly_detection" and contract.target not in columns:
                    errors.append(f"target {contract.target!r} not in dataset columns")
            except Exception:
                pass
        return errors

    def _prompt(self, description: str, dataset_path: str | Path | None) -> str:
        columns: list[str] = []
        dtypes: dict[str, str] = {}
        row_count = None
        if dataset_path:
            try:
                frame = pd.read_csv(dataset_path)
                columns = frame.columns.tolist()
                dtypes = {str(k): str(v) for k, v in frame.dtypes.items()}
                row_count = len(frame)
            except Exception:
                pass
        return json.dumps({
            "user_requirement": description,
            "dataset": {"path": str(dataset_path or ""), "columns": columns, "dtypes": dtypes, "row_count": row_count},
            "required_fields": ["domain", "capability_name", "task_type", "data_type", "target", "input_schema", "output_schema", "dataset", "metrics", "thresholds", "constraints", "latency_requirement_ms", "interpretability_requirement", "resource_constraint", "probability_output_required", "class_imbalance", "candidate_hints", "uncertainty", "confidence"],
            "allowed_task_types": ["binary_classification", "multiclass_classification", "regression", "anomaly_detection", "text_classification"],
            "allowed_data_types": ["tabular", "text", "image", "timeseries"],
            "rule": "All required fields must be present. Use empty arrays/objects or null only where appropriate. Do not copy schema examples as values.",
        }, ensure_ascii=False)

    def _to_spec(self, contract: RequirementContract, fallback: CapabilitySpec, description: str, dataset_path: str | Path | None) -> tuple[CapabilitySpec, list[str]]:
        corrections: list[str] = []
        target = "" if contract.task_type == "anomaly_detection" and not contract.target else contract.target
        features = list(contract.input_schema) or fallback.feature_columns
        if dataset_path:
            columns = pd.read_csv(dataset_path, nrows=0).columns.tolist()
            features = columns if not target else [c for c in columns if c != target]
        thresholds = contract.thresholds or fallback.metric_thresholds
        return CapabilitySpec(
            raw_description=description, domain=contract.domain, capability_name=contract.capability_name,
            task_type=contract.task_type, data_type=contract.data_type, target_column=target,
            feature_columns=features, input_schema=contract.input_schema or {c: "unknown" for c in features},
            output_schema=contract.output_schema, dataset_profile=contract.dataset, metrics=contract.metrics,
            metric_thresholds=thresholds, constraints=contract.constraints,
            latency_requirement_ms=contract.latency_requirement_ms,
            interpretability_requirement=contract.interpretability_requirement,
            resource_constraints=contract.resource_constraint,
            probability_output_required=contract.probability_output_required,
            class_imbalance=contract.class_imbalance, candidate_hints=contract.candidate_hints,
            uncertainty=contract.uncertainty, understanding_confidence=contract.confidence,
            candidate_algorithms=fallback.candidate_algorithms, dataset_path=str(dataset_path) if dataset_path else None,
        ), corrections

