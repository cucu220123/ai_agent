from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CapabilitySpec:
    raw_description: str
    capability_name: str = "客户流失预测"
    task_type: str = "binary_classification"
    target_column: str = "churn"
    feature_columns: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=lambda: ["roc_auc", "f1", "precision", "recall"])
    metric_thresholds: dict[str, float] = field(default_factory=lambda: {"roc_auc": 0.75})
    output_columns: list[str] = field(default_factory=lambda: ["prediction", "probability"])
    constraints: list[str] = field(default_factory=list)
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
            "report_json": self.report_json,
            "report_markdown": self.report_markdown,
        }
