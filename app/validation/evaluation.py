"""Trusted metric implementation, executed in the parent (never generated evaluate)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import metrics as sm
from sklearn.model_selection import train_test_split
from typing import Any


def split_frames(frame: pd.DataFrame, target: str, task_type: str):
    if not target:
        return frame.copy(), frame.copy()
    stratify = None
    if task_type in {"binary_classification", "multiclass_classification", "text_classification"}:
        counts = frame[target].value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = frame[target]
    return train_test_split(frame, test_size=.25, random_state=42, stratify=stratify)


def trusted_metrics(test: pd.DataFrame, prediction: dict[str, list], target: str, task_type: str) -> dict[str, float]:
    predicted = np.asarray(prediction["prediction"])
    if not target:
        scores = np.asarray(prediction.get("anomaly_score", []), dtype=float)
        # Descriptive diagnostics only. No unsupervised accuracy can be inferred here.
        return {"anomaly_rate": float(np.mean(predicted == 1)), "mean_anomaly_score": float(scores.mean()) if len(scores) else 0.0}
    y = test[target].to_numpy()
    if task_type == "regression":
        return {"mae": float(sm.mean_absolute_error(y, predicted)), "rmse": float(np.sqrt(sm.mean_squared_error(y, predicted))), "r2": float(sm.r2_score(y, predicted))}
    labels = sorted(pd.unique(y), key=str)
    binary = task_type in {"binary_classification", "anomaly_detection"} and len(labels) == 2
    kwargs = {"average": "binary", "pos_label": labels[-1]} if binary else {"average": "weighted"}
    result = {"accuracy": float(sm.accuracy_score(y, predicted)), "f1": float(sm.f1_score(y, predicted, zero_division=0, **kwargs)), "precision": float(sm.precision_score(y, predicted, zero_division=0, **kwargs)), "recall": float(sm.recall_score(y, predicted, zero_division=0, **kwargs)), "balanced_accuracy": float(sm.balanced_accuracy_score(y, predicted))}
    if binary and "probability" in prediction:
        probability = np.asarray(prediction["probability"], dtype=float)
        binary_y = (y == labels[-1]).astype(int)
        result.update(roc_auc=float(sm.roc_auc_score(binary_y, probability)), pr_auc=float(sm.average_precision_score(binary_y, probability)))
    return result


def validate_outputs(value: Any, rows: int, outputs: list[str]) -> dict[str, list]:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("predict must return a pandas DataFrame")
    if len(value) != rows:
        raise ValueError("prediction output shape does not match input rows")
    if not set(outputs).issubset(value.columns):
        raise ValueError(f"prediction output missing {sorted(set(outputs) - set(value.columns))}")
    if value.isna().any().any():
        raise ValueError("predictions contain NaN/missing values")
    for column in value.select_dtypes(include="number").columns:
        if not np.isfinite(value[column].to_numpy(dtype=float)).all():
            raise ValueError("predictions contain infinity")
    if "probability" in value:
        probability = value["probability"]
        if not pd.api.types.is_numeric_dtype(probability) or not probability.between(0, 1).all():
            raise ValueError("probability must be numeric and in [0,1]")
    return {str(k): v.tolist() for k, v in value.items()}

