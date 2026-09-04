from __future__ import annotations

import json

from app.models import AlgorithmPlan, CapabilitySpec


def render_algorithm(spec: CapabilitySpec, plan: AlgorithmPlan) -> str:
    """Render a safe, deterministic sklearn module constrained by the task contract."""
    feature_hint = json.dumps(spec.feature_columns, ensure_ascii=False)
    params = plan.hyperparameters
    if plan.algorithm_id.endswith("logistic_regression"):
        estimator = f"LogisticRegression(C={params.get('C', 1.0)!r}, max_iter={params.get('max_iter', 500)!r}, random_state=42)"
        scale = '    numeric_steps.append(("scale", StandardScaler()))\n'
    elif plan.algorithm_id.endswith("random_forest"):
        estimator = f"RandomForestClassifier(n_estimators={params.get('n_estimators', 180)!r}, max_depth={params.get('max_depth', 8)!r}, random_state=42, n_jobs=1, class_weight=\"balanced\")"
        scale = ""
    else:
        estimator = f"GradientBoostingClassifier(n_estimators={params.get('n_estimators', 120)!r}, learning_rate={params.get('learning_rate', 0.05)!r}, max_depth={params.get('max_depth', 3)!r}, random_state=42)"
        scale = ""
    return f'''"""Generated algorithm for: {spec.capability_name}."""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, balanced_accuracy_score, precision_recall_curve

FEATURE_HINT = {feature_hint}
ALGORITHM_NAME = {plan.algorithm_name!r}


def _build_pipeline(train_df: pd.DataFrame, target_col: str, config: Optional[dict] = None):
    config = config or {{}}
    X = train_df.drop(columns=[target_col])
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
{scale}    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    estimator = {estimator}
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def train(train_df: pd.DataFrame, target_col: str, config: Optional[dict] = None):
    if target_col not in train_df.columns:
        raise ValueError(f"target column '{{target_col}}' not found")
    if len(train_df) < 10:
        raise ValueError("at least 10 rows are required")
    model = _build_pipeline(train_df, target_col, config)
    model.fit(train_df.drop(columns=[target_col]), train_df[target_col].astype(int))
    model._ai_factory_threshold = float((config or {{}}).get("threshold", 0.5))
    return model


def predict(model, test_df: pd.DataFrame) -> pd.DataFrame:
    probability = model.predict_proba(test_df)[:, 1]
    prediction = (probability >= float(getattr(model, "_ai_factory_threshold", 0.5))).astype(int)
    return pd.DataFrame({{"prediction": prediction.astype(int), "probability": probability.astype(float)}})


def evaluate(model, test_df: pd.DataFrame, target_col: str) -> dict:
    if target_col not in test_df.columns:
        raise ValueError(f"target column '{{target_col}}' not found")
    y_true = test_df[target_col].astype(int)
    pred = predict(model, test_df.drop(columns=[target_col]))
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, pred["probability"])
    f1_curve = 2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-12)
    best_idx = int(np.nanargmax(f1_curve)) if len(f1_curve) else 0
    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) else 0.5
    return {{
        "roc_auc": float(roc_auc_score(y_true, pred["probability"])),
        "pr_auc": float(average_precision_score(y_true, pred["probability"])),
        "f1": float(f1_score(y_true, pred["prediction"], zero_division=0)),
        "precision": float(precision_score(y_true, pred["prediction"], zero_division=0)),
        "recall": float(recall_score(y_true, pred["prediction"], zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred["prediction"])),
        "best_f1": float(f1_curve[best_idx]) if len(f1_curve) else 0.0,
        "best_threshold": best_threshold,
    }}
'''
