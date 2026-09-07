from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.parser_agent import ParserAgent
from app.llm.contracts import extract_json_object
from app.models import CapabilitySpec


class RequirementContract(BaseModel):
    """Strict LLM contract. Unknown fields are ignored; semantic checks happen below."""

    model_config = ConfigDict(extra="ignore")
    domain: str = "general"
    capability_name: str = "算法能力"
    task_type: str = "binary_classification"
    data_type: str = "tabular"
    target: str = "churn"
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    latency_requirement_ms: float | None = None
    interpretability_requirement: str | None = None
    resource_constraint: dict[str, Any] = Field(default_factory=dict)
    probability_output_required: bool = False
    class_imbalance: dict[str, Any] = Field(default_factory=dict)
    candidate_hints: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class RequirementUnderstandingAgent:
    """LLM-first requirement understanding with deterministic semantic correction."""

    def __init__(self, llm=None, provider_name: str = "mock"):
        self.llm = llm
        self.provider_name = provider_name
        self.fallback = ParserAgent()

    def run(self, description: str, dataset_path: str | Path | None = None) -> tuple[CapabilitySpec, dict[str, Any]]:
        fallback = self.fallback.run(description, str(dataset_path) if dataset_path else None)
        raw = ""
        trace: dict[str, Any] = {"provider": self.provider_name, "status": "fallback", "schema_valid": False, "semantic_corrections": []}
        if self.llm is not None and self.provider_name != "mock":
            prompt = self._prompt(description, dataset_path)
            try:
                raw = self.llm.complete(
                    "你是 RequirementUnderstandingAgent。只返回一个符合给定字段的 JSON 对象，不要 Markdown，不要解释。",
                    prompt,
                ) or ""
                parsed = extract_json_object(raw)
                contract = RequirementContract.model_validate(parsed or {})
                spec, corrections = self._to_spec(contract, fallback, description, dataset_path)
                trace.update({"status": "ok", "schema_valid": True, "semantic_corrections": corrections, "response_preview": raw[:1000]})
                return spec, trace
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                trace["error"] = f"{type(exc).__name__}: {exc}"[:1000]
            except Exception as exc:
                trace["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        return fallback, trace

    def _prompt(self, description: str, dataset_path: str | Path | None) -> str:
        columns: list[str] = []
        dtypes: dict[str, str] = {}
        if dataset_path:
            try:
                frame = pd.read_csv(dataset_path, nrows=50)
                columns = frame.columns.tolist()
                dtypes = {str(k): str(v) for k, v in frame.dtypes.items()}
            except Exception:
                pass
        schema = {
            "domain": "string",
            "capability_name": "string",
            "task_type": "binary_classification|multiclass_classification|regression|anomaly_detection|text_classification",
            "data_type": "tabular|text|image|timeseries",
            "target": "string or empty for unsupervised",
            "input_schema": {"column": "type"},
            "output_schema": {"column": "type"},
            "dataset": {"path": "string", "columns": ["string"], "row_count": "number"},
            "metrics": ["string"],
            "thresholds": {"metric": "number"},
            "constraints": ["string"],
            "latency_requirement_ms": "number or null",
            "interpretability_requirement": "string or null",
            "resource_constraint": {"key": "value"},
            "probability_output_required": "boolean",
            "class_imbalance": {"positive_rate": "number or null", "description": "string"},
            "candidate_hints": ["string"],
            "uncertainty": ["string"],
            "confidence": "number in [0,1]",
        }
        return json.dumps({"user_requirement": description, "dataset_columns": columns, "dataset_dtypes": dtypes, "output_schema": schema}, ensure_ascii=False)

    def _to_spec(self, contract: RequirementContract, fallback: CapabilitySpec, description: str, dataset_path: str | Path | None) -> tuple[CapabilitySpec, list[str]]:
        corrections: list[str] = []
        allowed_tasks = {"binary_classification", "multiclass_classification", "regression", "anomaly_detection", "text_classification"}
        task_type = contract.task_type if contract.task_type in allowed_tasks else fallback.task_type
        if task_type != contract.task_type:
            corrections.append("task_type corrected to supported plugin")
        target = contract.target.strip() if isinstance(contract.target, str) else fallback.target_column
        if task_type == "anomaly_detection":
            target = ""
        elif dataset_path:
            try:
                columns = pd.read_csv(dataset_path, nrows=0).columns.tolist()
                if target not in columns:
                    corrections.append("target corrected to dataset-compatible target")
                    target = fallback.target_column if fallback.target_column in columns else (columns[-1] if columns else target)
            except Exception:
                pass
        features = list(contract.input_schema) or fallback.feature_columns
        if dataset_path:
            try:
                columns = pd.read_csv(dataset_path, nrows=0).columns.tolist()
                if task_type == "anomaly_detection":
                    features = columns
                elif target in columns:
                    features = [c for c in columns if c != target]
            except Exception:
                pass
        metrics = contract.metrics or fallback.metrics
        thresholds = {str(k): float(v) for k, v in contract.thresholds.items() if isinstance(v, (int, float))}
        if not thresholds:
            thresholds = fallback.metric_thresholds
            corrections.append("metric thresholds filled from deterministic fallback")
        output_schema = contract.output_schema or ({"prediction": "int", "probability": "float"} if task_type in {"binary_classification", "text_classification"} else {"prediction": "float"})
        domain = contract.domain or fallback.domain
        capability_name = contract.capability_name or fallback.capability_name
        if domain == "general" and fallback.domain != "general":
            domain = fallback.domain
            corrections.append("domain filled from validated fallback context")
        if capability_name == "算法能力" and fallback.capability_name != "算法能力":
            capability_name = fallback.capability_name
            corrections.append("capability name filled from validated fallback context")
        return CapabilitySpec(
            raw_description=description,
            domain=domain,
            capability_name=capability_name,
            task_type=task_type,
            data_type=contract.data_type or fallback.data_type,
            target_column=target,
            feature_columns=features,
            input_schema=contract.input_schema or {c: "unknown" for c in features},
            output_schema=output_schema,
            dataset_profile=contract.dataset,
            metrics=metrics,
            metric_thresholds=thresholds,
            constraints=contract.constraints or fallback.constraints,
            latency_requirement_ms=contract.latency_requirement_ms,
            interpretability_requirement=contract.interpretability_requirement,
            resource_constraints=contract.resource_constraint,
            probability_output_required=contract.probability_output_required or fallback.probability_output_required,
            class_imbalance=contract.class_imbalance,
            candidate_hints=contract.candidate_hints,
            uncertainty=contract.uncertainty,
            understanding_confidence=max(0.0, min(1.0, contract.confidence)),
            candidate_algorithms=fallback.candidate_algorithms,
            dataset_path=str(dataset_path) if dataset_path else fallback.dataset_path,
        ), corrections
