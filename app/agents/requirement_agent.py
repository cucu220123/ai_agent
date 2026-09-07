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
from app.validation.dataset import profile_dataset
from app.metrics.registry import METRIC_REGISTRY
from app.plugins.registry import DEFAULT_REGISTRY
from app.llm.secrets import sanitize


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
        if not description.strip():
            raise ValueError("能力描述不能为空")
        fallback = CapabilitySpec(raw_description=description)
        trace: dict[str, Any] = {"provider": self.provider_name, "status": "fallback", "schema_valid": False, "semantic_valid": False, "attempts": []}
        if self.llm is None or self.provider_name == "mock":
            trace["fallback_reason"] = "mock_or_missing_provider"
            return self.fallback.run(description, str(dataset_path) if dataset_path else None), trace

        prompt = self._prompt(description, dataset_path)
        previous = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                if attempt > 1:
                    prompt = json.dumps({"instruction": "Repair the previous JSON. Return only a complete object matching the schema.", "previous_output": previous[:8000], "validation_error": trace["attempts"][-1].get("error"), "original_request": json.loads(self._prompt(description, dataset_path))}, ensure_ascii=False)
                raw = self.llm.complete("你是 RequirementUnderstandingAgent。理解用户的领域、任务和约束。只返回严格 JSON，不要 Markdown。统计事实以数据画像为准；不确定信息列入 uncertainty。", prompt, purpose="requirement", generation_config={"json_schema": RequirementContract.model_json_schema()}) or ""
                previous = raw
                parsed = self._normalize_candidate(extract_json_object(raw))
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
        return self.fallback.run(description, str(dataset_path) if dataset_path else None), sanitize(trace)

    @staticmethod
    def _normalize_candidate(parsed: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            return parsed
        value = dict(parsed)
        # Only mechanical type repairs are allowed; semantic fields remain required and are never invented here.
        for key in ("input_schema", "output_schema", "resource_constraint"):
            if isinstance(value.get(key), str):
                try:
                    value[key] = json.loads(value[key])
                except json.JSONDecodeError:
                    pass
        for key in ("input_schema", "output_schema"):
            if isinstance(value.get(key), dict) and isinstance(value[key].get("fields"), list):
                value[key] = {str(item.get("name")): str(item.get("type", "unknown")) for item in value[key]["fields"] if isinstance(item, dict) and item.get("name")}
        if isinstance(value.get("interpretability_requirement"), bool):
            value["interpretability_requirement"] = "high" if value["interpretability_requirement"] else "none"
        for key in ("resource_constraint", "class_imbalance"):
            if value.get(key) is None:
                value[key] = {}
        for key in ("constraints", "uncertainty", "candidate_hints"):
            if isinstance(value.get(key), dict):
                value[key] = [f"{k}={v}" for k, v in value[key].items()]
            elif value.get(key) is None:
                value[key] = []
        if isinstance(value.get("class_imbalance"), bool):
            value["class_imbalance"] = {"is_imbalanced": value["class_imbalance"]}
        if value.get("confidence") is None:
            value["confidence"] = 0.0
        return value

    def _quality_gate(self, contract: RequirementContract, description: str, dataset_path: str | Path | None) -> list[str]:
        errors = []
        if contract.confidence < 0.45:
            errors.append("confidence below 0.45")
        lowered = description.lower()
        required_metrics = []
        metric_text = lowered.replace("pr-auc", "pr_auc").replace("pr_auc", "average precision")
        for alias, canonical in (("roc-auc", "roc_auc"), ("auc", "roc_auc"), ("average precision", "pr_auc"), ("f1", "f1"), ("mae", "mae"), ("rmse", "rmse"), ("r2", "r2")):
            if alias in metric_text and canonical not in required_metrics:
                required_metrics.append(canonical)
        missing = set(required_metrics) - set(contract.metrics)
        if missing:
            errors.append(f"metrics omit user requirements: {sorted(missing)}")
        for metric in set(contract.metrics) | set(contract.thresholds):
            definition = METRIC_REGISTRY.metrics.get(metric)
            if definition is None or contract.task_type not in definition.tasks:
                errors.append(f"metric {metric} is not registered for {contract.task_type}")
        for metric, threshold in contract.thresholds.items():
            if metric in {"roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall", "balanced_accuracy"} and not 0 <= threshold <= 1:
                errors.append(f"threshold for {metric} must be in [0,1]")
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
            "json_schema": RequirementContract.model_json_schema(),
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
        thresholds = contract.thresholds
        profile = profile_dataset(dataset_path, target) if dataset_path else contract.dataset
        balance = dict(contract.class_imbalance)
        if "positive_rate" in profile:
            balance.update(positive_rate=profile["positive_rate"], is_imbalanced=profile["minority_rate"] < 0.25)
        candidates = [p.id for p in DEFAULT_REGISTRY.algorithms_for(contract.task_type) if p.id != "dummy_classifier"]
        outputs = contract.output_schema or {name: "float" for name in DEFAULT_REGISTRY.tasks[contract.task_type].output_columns}
        return CapabilitySpec(
            raw_description=description, domain=contract.domain, capability_name=contract.capability_name,
            task_type=contract.task_type, data_type=contract.data_type, target_column=target,
            feature_columns=features, input_schema=contract.input_schema or {c: "unknown" for c in features},
            output_schema=outputs, output_columns=list(outputs), dataset_profile=profile, metrics=contract.metrics,
            metric_thresholds=thresholds, constraints=contract.constraints,
            latency_requirement_ms=contract.latency_requirement_ms,
            interpretability_requirement=contract.interpretability_requirement,
            resource_constraints=contract.resource_constraint,
            probability_output_required=contract.probability_output_required,
            class_imbalance=balance, candidate_hints=contract.candidate_hints,
            uncertainty=contract.uncertainty, understanding_confidence=contract.confidence,
            candidate_algorithms=candidates, dataset_path=str(dataset_path) if dataset_path else None,
        ), corrections
