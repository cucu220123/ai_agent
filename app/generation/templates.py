from __future__ import annotations

import json

from app.models import AlgorithmPlan, CapabilitySpec


def render_algorithm(spec: CapabilitySpec, plan: AlgorithmPlan) -> str:
    from app.plugins.registry import DEFAULT_REGISTRY
    key = (plan.base_algorithm_id or plan.algorithm_id).removeprefix("algorithm_").split("__")[0]
    plugin = DEFAULT_REGISTRY.algorithms.get(key)
    if plugin and plugin.renderer:
        source = plugin.renderer(spec, plan)
    else:
        if key not in {"logistic_regression", "random_forest", "gradient_boosting", "random_forest_regressor", "isolation_forest", "tfidf_logistic_regression"}:
            raise ValueError(f"plugin {key} needs a registered renderer for deterministic fallback")
        source = _render_algorithm(spec, plan)
    if "def metadata(" not in source:
        metadata = {"protocol_version": "1.1", "algorithm": plan.algorithm_name, "rationale": plan.rationale, "evidence_ids": plan.evidence_ids, "parameters": plan.hyperparameters, "source": "deterministic_template"}
        source += "\n\ndef metadata():\n    return " + repr(metadata) + "\n"
    if spec.task_type in {"binary_classification", "text_classification", "multiclass_classification"} and "def predict_proba(" not in source:
        source += "\n\ndef predict_proba(model, test_df):\n    return predict(model, test_df)['probability'].to_numpy()\n"
    if not (plugin and plugin.renderer):
        # All built-in estimators consume the worker's seed; variance checks
        # must exercise independent seeds instead of repeating a hardcoded 42.
        source = source.replace("random_state=42", "random_state=(config or {}).get('random_state', 42)")
    return source


def _render_algorithm(spec: CapabilitySpec, plan: AlgorithmPlan) -> str:
    """Render a safe, deterministic sklearn module constrained by the task contract."""
    if spec.task_type == "regression":
        return render_regression_algorithm(spec, plan)
    if spec.task_type == "anomaly_detection":
        return render_anomaly_algorithm(spec, plan)
    if spec.task_type == "text_classification":
        return render_text_algorithm(spec, plan)
    feature_hint = json.dumps(spec.feature_columns, ensure_ascii=False)
    params = plan.hyperparameters
    algorithm_key = plan.base_algorithm_id or plan.algorithm_id
    numeric_imputer = params.get("numeric_imputer", "median")
    scaler_kind = params.get("scaler", "none")
    if algorithm_key.endswith("logistic_regression"):
        estimator = f"LogisticRegression(C={params.get('C', 1.0)!r}, max_iter={params.get('max_iter', 500)!r}, class_weight={params.get('class_weight')!r}, random_state=42)"
        scale = '    numeric_steps.append(("scale", RobustScaler()))\n' if scaler_kind == "robust" else ('    numeric_steps.append(("scale", StandardScaler()))\n' if scaler_kind == "standard" else "")
    elif algorithm_key.endswith("random_forest"):
        estimator = f"RandomForestClassifier(n_estimators={params.get('n_estimators', 180)!r}, max_depth={params.get('max_depth', 8)!r}, min_samples_leaf={params.get('min_samples_leaf', 1)!r}, random_state=42, n_jobs=1, class_weight={params.get('class_weight', 'balanced')!r})"
        scale = ""
    else:
        estimator = f"GradientBoostingClassifier(n_estimators={params.get('n_estimators', 120)!r}, learning_rate={params.get('learning_rate', 0.05)!r}, max_depth={params.get('max_depth', 3)!r}, random_state=42)"
        scale = ""
    return f'''"""Generated tabular classification algorithm."""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
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
    numeric_steps = [("impute", SimpleImputer(strategy={numeric_imputer!r}))]
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
    model._ai_factory_threshold = float((config or {{}}).get("threshold", {params.get('threshold', 0.5)!r}))
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


def render_regression_algorithm(spec: CapabilitySpec, plan: AlgorithmPlan) -> str:
    params = plan.hyperparameters
    return f'''"""Generated regression algorithm."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ALGORITHM_NAME = {plan.algorithm_name!r}

def _build_pipeline(train_df: pd.DataFrame, target_col: str, config=None):
    X = train_df.drop(columns=[target_col])
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    estimator = RandomForestRegressor(n_estimators={params.get('n_estimators', 160)!r}, max_depth={params.get('max_depth', 10)!r}, min_samples_leaf={params.get('min_samples_leaf', 1)!r}, random_state=42, n_jobs=1)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

def train(train_df: pd.DataFrame, target_col: str, config: Optional[dict] = None):
    if target_col not in train_df.columns: raise ValueError(f"target column '{{target_col}}' not found")
    model = _build_pipeline(train_df, target_col, config)
    model.fit(train_df.drop(columns=[target_col]), train_df[target_col].astype(float))
    return model

def predict(model, test_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({{"prediction": model.predict(test_df).astype(float)}})

def evaluate(model, test_df: pd.DataFrame, target_col: str) -> dict:
    y_true = test_df[target_col].astype(float)
    prediction = predict(model, test_df.drop(columns=[target_col]))["prediction"]
    return {{"rmse": float(np.sqrt(mean_squared_error(y_true, prediction))), "mae": float(mean_absolute_error(y_true, prediction)), "r2": float(r2_score(y_true, prediction))}}
'''


def render_anomaly_algorithm(spec: CapabilitySpec, plan: AlgorithmPlan) -> str:
    params = plan.hyperparameters
    return f'''"""Generated anomaly detection algorithm."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
ALGORITHM_NAME = {plan.algorithm_name!r}

def train(train_df: pd.DataFrame, target_col: str, config: Optional[dict] = None):
    X = train_df.drop(columns=[target_col]) if target_col and target_col in train_df.columns else train_df.copy()
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric: raise ValueError("anomaly detection requires numeric features")
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", IsolationForest(n_estimators={params.get('n_estimators', 160)!r}, contamination={params.get('contamination', 'auto')!r}, random_state=42, n_jobs=1))])
    model.fit(X[numeric])
    model._ai_factory_features = numeric
    return model

def predict(model, test_df: pd.DataFrame) -> pd.DataFrame:
    X = test_df[getattr(model, "_ai_factory_features", test_df.select_dtypes(include=[np.number]).columns.tolist())]
    raw = model.predict(X)
    return pd.DataFrame({{"prediction": (raw == -1).astype(int), "anomaly_score": (-model.decision_function(X)).astype(float)}})

def evaluate(model, test_df: pd.DataFrame, target_col: str) -> dict:
    pred = predict(model, test_df.drop(columns=[target_col]) if target_col and target_col in test_df.columns else test_df)
    if target_col and target_col in test_df.columns:
        from sklearn.metrics import f1_score, precision_score, recall_score
        y_true = test_df[target_col].astype(int)
        return {{"f1": float(f1_score(y_true, pred["prediction"], zero_division=0)), "precision": float(precision_score(y_true, pred["prediction"], zero_division=0)), "recall": float(recall_score(y_true, pred["prediction"], zero_division=0))}}
    return {{"anomaly_rate": float(pred["prediction"].mean()), "mean_anomaly_score": float(pred["anomaly_score"].mean())}}
'''


def render_text_algorithm(spec: CapabilitySpec, plan: AlgorithmPlan) -> str:
    text_column = spec.feature_columns[0] if spec.feature_columns else "text"
    params = plan.hyperparameters
    return f'''"""Generated text classification algorithm."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
TEXT_COLUMN = {text_column!r}
ALGORITHM_NAME = {plan.algorithm_name!r}

def train(train_df: pd.DataFrame, target_col: str, config: Optional[dict] = None):
    if TEXT_COLUMN not in train_df.columns: raise ValueError(f"text column '{{TEXT_COLUMN}}' not found")
    model = Pipeline([("tfidf", TfidfVectorizer(max_features={params.get('max_features', 5000)!r}, ngram_range=(1, {params.get('ngram_max', 2)!r}), min_df=1)), ("model", LogisticRegression(C={params.get('C', 1.0)!r}, max_iter=500, random_state=42))])
    model.fit(train_df[TEXT_COLUMN].fillna("").astype(str), train_df[target_col])
    return model

def predict(model, test_df: pd.DataFrame) -> pd.DataFrame:
    text = test_df[TEXT_COLUMN].fillna("").astype(str)
    prediction = model.predict(text)
    probabilities = model.predict_proba(text)
    probability = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities.max(axis=1)
    return pd.DataFrame({{"prediction": prediction, "probability": probability.astype(float)}})

def evaluate(model, test_df: pd.DataFrame, target_col: str) -> dict:
    y_true = test_df[target_col]
    pred = predict(model, test_df.drop(columns=[target_col]))["prediction"]
    return {{"accuracy": float(accuracy_score(y_true, pred)), "f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)), "precision": float(precision_score(y_true, pred, average="weighted", zero_division=0)), "recall": float(recall_score(y_true, pred, average="weighted", zero_division=0))}}
'''
