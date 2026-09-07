from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CapabilitySpec:
    raw_description: str
    domain: str = "customer_churn"
    capability_name: str = "客户流失预测"
    task_type: str = "binary_classification"
    data_type: str = "tabular"
    target_column: str = "churn"
    feature_columns: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=lambda: {"prediction": "int", "probability": "float"})
    dataset_profile: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=lambda: ["roc_auc", "f1", "precision", "recall"])
    metric_thresholds: dict[str, float] = field(default_factory=lambda: {"roc_auc": 0.75})
    output_columns: list[str] = field(default_factory=lambda: ["prediction", "probability"])
    constraints: list[str] = field(default_factory=list)
    latency_requirement_ms: float | None = None
    interpretability_requirement: str | None = None
    resource_constraints: dict[str, Any] = field(default_factory=dict)
    probability_output_required: bool = True
    class_imbalance: dict[str, Any] = field(default_factory=dict)
    candidate_hints: list[str] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    understanding_confidence: float = 0.0
    candidate_algorithms: list[str] = field(default_factory=list)
    dataset_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeContext:
    capabilities: list[dict[str, Any]] = field(default_factory=list)
    algorithms: list[dict[str, Any]] = field(default_factory=list)
    experiences: list[dict[str, Any]] = field(default_factory=list)
    retrieval_query: str = ""
    graph_evidence: dict[str, Any] = field(default_factory=dict)
    semantic_evidence: list[dict[str, Any]] = field(default_factory=list)
    historical_cases: list[dict[str, Any]] = field(default_factory=list)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)
    planning_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlgorithmPlan:
    algorithm_id: str
    algorithm_name: str
    rationale: str
    preprocessing: list[str]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    expected_metrics: dict[str, float] = field(default_factory=dict)
    priority: int = 0
    base_algorithm_id: str | None = None
    preprocessing_variant: str = "default"
    config_variant: str = "default"
    search_score: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    status: str
    algorithm: str
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    repair_round: int = 0
    task_type: str = "binary_classification"
    dataset_profile: dict[str, Any] = field(default_factory=dict)
    resource_usage: dict[str, Any] = field(default_factory=dict)
    failure_type: str | None = None
    root_cause: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowResult:
    run_id: str
    spec: CapabilitySpec
    knowledge: KnowledgeContext
    plans: list[AlgorithmPlan]
    selected_plan: AlgorithmPlan | None
    generated_files: list[str]
    validation: ValidationResult | None
    repair_history: list[dict[str, Any]]
    candidate_results: list[dict[str, Any]] = field(default_factory=list)
    llm_trace: dict[str, Any] = field(default_factory=dict)
    search_trace: dict[str, Any] = field(default_factory=dict)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    report_json: str | None = None
    report_markdown: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "spec": self.spec.to_dict(),
            "knowledge": self.knowledge.to_dict(),
            "plans": [p.to_dict() for p in self.plans],
            "selected_plan": self.selected_plan.to_dict() if self.selected_plan else None,
            "generated_files": self.generated_files,
            "validation": self.validation.to_dict() if self.validation else None,
            "repair_history": self.repair_history,
            "candidate_results": self.candidate_results,
            "llm_trace": self.llm_trace,
            "search_trace": self.search_trace,
            "event_log": self.event_log,
            "report_json": self.report_json,
            "report_markdown": self.report_markdown,
        }
