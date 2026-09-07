from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    maximize: bool
    tasks: tuple[str, ...]


class MetricRegistry:
    def __init__(self):
        self.metrics = {
            "roc_auc": MetricDefinition("roc_auc", True, ("binary_classification",)),
            "pr_auc": MetricDefinition("pr_auc", True, ("binary_classification",)),
            "f1": MetricDefinition("f1", True, ("binary_classification", "multiclass_classification", "text_classification", "anomaly_detection")),
            "accuracy": MetricDefinition("accuracy", True, ("binary_classification", "multiclass_classification", "text_classification")),
            "precision": MetricDefinition("precision", True, ("binary_classification", "text_classification", "anomaly_detection")),
            "recall": MetricDefinition("recall", True, ("binary_classification", "text_classification", "anomaly_detection")),
            "balanced_accuracy": MetricDefinition("balanced_accuracy", True, ("binary_classification",)),
            "r2": MetricDefinition("r2", True, ("regression",)),
            "mae": MetricDefinition("mae", False, ("regression",)),
            "rmse": MetricDefinition("rmse", False, ("regression",)),
            "runtime_seconds": MetricDefinition("runtime_seconds", False, ("anomaly_detection",)),
        }
        self.defaults = {
            "binary_classification": "roc_auc",
            "multiclass_classification": "f1",
            "text_classification": "f1",
            "regression": "r2",
            "anomaly_detection": "f1",
        }

    def primary(self, task_type: str, requested: list[str] | None = None, has_target: bool = True) -> str:
        if task_type == "anomaly_detection" and not has_target:
            return "runtime_seconds"
        for metric in requested or []:
            definition = self.metrics.get(metric)
            if definition and task_type in definition.tasks:
                return metric
        return self.defaults.get(task_type, "runtime_seconds")

    def is_maximize(self, metric: str) -> bool:
        return self.metrics.get(metric, MetricDefinition(metric, True, ())).maximize

    def passes(self, metric: str, actual: float | None, threshold: float) -> bool:
        if actual is None:
            return False
        return actual >= threshold if self.is_maximize(metric) else actual <= threshold

    def ranking_score(self, task_type: str, metrics: dict[str, float], runtime_seconds: float, requested: list[str] | None = None, has_target: bool = True) -> tuple[float, str]:
        primary = self.primary(task_type, requested, has_target)
        value = runtime_seconds if primary == "runtime_seconds" else metrics.get(primary)
        if value is None:
            return (-float("inf"), primary)
        return (float(value) if self.is_maximize(primary) else -float(value), primary)


METRIC_REGISTRY = MetricRegistry()

