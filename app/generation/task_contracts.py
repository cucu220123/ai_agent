from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskCodeContract:
    task_type: str
    required_outputs: tuple[str, ...]
    optional_outputs: tuple[str, ...]
    metrics: tuple[str, ...]
    prompt_requirements: tuple[str, ...]
    primary_metric: str
    quality_metric: str


TASK_CODE_CONTRACTS: dict[str, TaskCodeContract] = {
    "binary_classification": TaskCodeContract("binary_classification", ("prediction", "probability"), (), ("roc_auc", "pr_auc", "f1", "precision", "recall"), ("probability must be in [0,1]", "evaluate returns roc_auc/f1/precision/recall"), "roc_auc", "supervised"),
    "regression": TaskCodeContract("regression", ("prediction",), (), ("r2", "mae", "rmse"), ("do not output probability", "evaluate returns r2/mae/rmse"), "r2", "supervised"),
    "text_classification": TaskCodeContract("text_classification", ("prediction",), ("probability",), ("f1", "accuracy"), ("use text column and TF-IDF", "probability is optional when classifier supports it"), "f1", "supervised"),
    "anomaly_detection": TaskCodeContract("anomaly_detection", ("prediction", "anomaly_score"), (), ("f1", "precision", "recall", "unsupervised_proxy_score"), ("return anomaly_score", "without target report unsupervised_proxy_score, not runtime as quality"), "unsupervised_proxy_score", "unsupervised_proxy"),
}


def get_task_contract(task_type: str) -> TaskCodeContract:
    return TASK_CODE_CONTRACTS.get(task_type, TASK_CODE_CONTRACTS["binary_classification"])


def build_codegen_prompt(spec: Any, plan: Any) -> str:
    contract = get_task_contract(spec.task_type)
    outputs = ", ".join(contract.required_outputs + contract.optional_outputs)
    metrics = ", ".join(contract.metrics)
    return (
        f"Task type: {contract.task_type}. Required prediction outputs: {outputs}. Metrics: {metrics}. "
        f"Target column is passed dynamically as target_col='{spec.target_column}'. Features: {spec.feature_columns}. "
        f"Algorithm: {plan.algorithm_name}; parameters: {plan.hyperparameters}. "
        "Exact API: train(train_df, target_col, config=None), predict(model, test_df), evaluate(model, test_df, target_col). "
        + " ".join(contract.prompt_requirements)
        + " Never hard-code a target column in predict/evaluate. Never pass prediction threshold to an estimator constructor. "
        "Only use pandas, numpy and scikit-learn; no files, network or system calls."
    )

