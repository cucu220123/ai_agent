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
    preprocessing: list[str] = field(default_factory=list)
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    supported_metrics: list[str] = field(default_factory=list)
    codegen_hints: dict[str, Any] = field(default_factory=dict)


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
            "algorithms": [{"id": p.id, "name": p.name, "task_types": p.task_types, "description": p.description, "resource_profile": p.resource_profile, "default_params": p.default_params, "preprocessing": p.preprocessing, "search_space": p.search_space, "supported_metrics": p.supported_metrics, "codegen_hints": p.codegen_hints} for p in self.algorithms.values()],
        }


def build_default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_task(TaskPlugin("binary_classification", "二分类", "binary", ["roc_auc", "pr_auc", "f1", "balanced_accuracy"], ["prediction", "probability"], ["logistic_regression", "random_forest", "gradient_boosting"]))
    registry.register_task(TaskPlugin("text_classification", "文本分类", "binary_or_multiclass", ["f1", "accuracy"], ["prediction", "probability"], ["tfidf_logistic_regression"]))
    registry.register_task(TaskPlugin("regression", "回归", "continuous", ["rmse", "mae", "r2"], ["prediction"], ["random_forest_regressor"]))
    registry.register_task(TaskPlugin("anomaly_detection", "异常检测", "unsupervised", ["precision", "recall", "f1"], ["prediction", "anomaly_score"], ["isolation_forest"]))
    registry.register_algorithm(AlgorithmPlugin("logistic_regression", "Logistic Regression", ["binary_classification"], "线性、可解释的二分类基线", "low", {"C": 1.0, "max_iter": 500}, ["imputation", "one_hot", "scaling"], {"C": [0.3, 1.0, 3.0], "class_weight": [None, "balanced"], "threshold": [0.4, 0.5]}, ["roc_auc", "pr_auc", "f1"], {"protocol": "tabular_binary"}))
    registry.register_algorithm(AlgorithmPlugin("random_forest", "Random Forest", ["binary_classification"], "适合混合特征和非线性关系", "medium", {"n_estimators": 180, "max_depth": 8, "random_state": 42}, ["imputation", "one_hot"], {"n_estimators": [80, 180], "max_depth": [6, 10], "min_samples_leaf": [1, 2], "class_weight": [None, "balanced"]}, ["roc_auc", "f1"], {"protocol": "tabular_binary"}))
    registry.register_algorithm(AlgorithmPlugin("gradient_boosting", "Gradient Boosting", ["binary_classification"], "通常具有较好的排序能力", "medium", {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}, ["imputation", "one_hot"], {"n_estimators": [80, 120], "learning_rate": [0.03, 0.05], "max_depth": [2, 3]}, ["roc_auc", "f1"], {"protocol": "tabular_binary"}))
    registry.register_algorithm(AlgorithmPlugin("random_forest_regressor", "Random Forest Regressor", ["regression"], "稳健的非线性回归", "medium", {"n_estimators": 160, "max_depth": 10, "random_state": 42}, ["imputation", "one_hot"], {"n_estimators": [80, 160], "max_depth": [8, 12], "min_samples_leaf": [1, 2]}, ["r2", "mae", "rmse"], {"protocol": "tabular_regression"}))
    registry.register_algorithm(AlgorithmPlugin("isolation_forest", "Isolation Forest", ["anomaly_detection"], "无监督异常检测", "medium", {"n_estimators": 160, "contamination": "auto", "random_state": 42}, ["imputation", "scaling"], {"n_estimators": [120, 200], "contamination": ["auto", 0.05, 0.1]}, ["f1", "precision", "recall"], {"protocol": "anomaly"}))
    registry.register_algorithm(AlgorithmPlugin("tfidf_logistic_regression", "TF-IDF Logistic Regression", ["text_classification"], "轻量、可解释的文本分类基线", "low", {"max_features": 5000, "ngram_range": [1, 2], "max_iter": 500}, ["text_fillna", "tfidf"], {"C": [0.4, 1.0], "max_features": [3000, 5000], "ngram_max": [1, 2]}, ["f1", "accuracy"], {"protocol": "text"}))
    registry.register_algorithm(AlgorithmPlugin("dummy_classifier", "Dummy Classifier", ["binary_classification", "multiclass_classification"], "test plugin", "low", {"strategy": "prior"}, ["none"], {"strategy": ["prior", "most_frequent"]}, ["f1", "accuracy"], {"protocol": "tabular_binary"}))
    return registry


DEFAULT_REGISTRY = build_default_registry()
