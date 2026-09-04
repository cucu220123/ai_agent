from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


MetricFn = Callable[[Any, Any], float]


@dataclass
class AlgorithmPlugin:
    id: str
    name: str
    task_types: list[str]
    description: str
    resource_profile: str = "medium"
    default_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlugin:
    id: str
    name: str
    target_kind: str
    default_metrics: list[str]
    output_columns: list[str]
    algorithms: list[str]


class PluginRegistry:
    def __init__(self):
        self.algorithms: dict[str, AlgorithmPlugin] = {}
        self.tasks: dict[str, TaskPlugin] = {}

    def register_algorithm(self, plugin: AlgorithmPlugin) -> None:
        self.algorithms[plugin.id] = plugin

    def register_task(self, plugin: TaskPlugin) -> None:
        self.tasks[plugin.id] = plugin

    def algorithms_for(self, task_type: str) -> list[AlgorithmPlugin]:
        return [p for p in self.algorithms.values() if task_type in p.task_types]

    def describe(self) -> dict[str, Any]:
        return {
            "tasks": [{"id": p.id, "name": p.name, "target_kind": p.target_kind, "default_metrics": p.default_metrics, "output_columns": p.output_columns, "algorithms": p.algorithms} for p in self.tasks.values()],
            "algorithms": [{"id": p.id, "name": p.name, "task_types": p.task_types, "description": p.description, "resource_profile": p.resource_profile, "default_params": p.default_params} for p in self.algorithms.values()],
        }


def build_default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_task(TaskPlugin("binary_classification", "二分类", "binary", ["roc_auc", "pr_auc", "f1", "balanced_accuracy"], ["prediction", "probability"], ["logistic_regression", "random_forest", "gradient_boosting"]))
    registry.register_task(TaskPlugin("regression", "回归", "continuous", ["rmse", "mae", "r2"], ["prediction"], []))
    registry.register_task(TaskPlugin("anomaly_detection", "异常检测", "unsupervised", ["precision", "recall", "f1"], ["prediction", "anomaly_score"], []))
    registry.register_algorithm(AlgorithmPlugin("logistic_regression", "Logistic Regression", ["binary_classification"], "线性、可解释的二分类基线", "low", {"C": 1.0, "max_iter": 500}))
    registry.register_algorithm(AlgorithmPlugin("random_forest", "Random Forest", ["binary_classification"], "适合混合特征和非线性关系", "medium", {"n_estimators": 180, "max_depth": 8, "random_state": 42}))
    registry.register_algorithm(AlgorithmPlugin("gradient_boosting", "Gradient Boosting", ["binary_classification"], "通常具有较好的排序能力", "medium", {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}))
    return registry


DEFAULT_REGISTRY = build_default_registry()

