from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


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
    "anomaly_detection": TaskCodeContract("anomaly_detection", ("prediction", "anomaly_score"), (), ("f1", "precision", "recall", "anomaly_rate", "mean_anomaly_score"), ("return anomaly_score and prediction=1 for anomalies", "without a target return only descriptive anomaly_rate and mean_anomaly_score; never invent quality metrics"), "anomaly_rate", "descriptive_only_without_labels"),
}


def get_task_contract(task_type: str) -> TaskCodeContract:
    if task_type not in TASK_CODE_CONTRACTS:
        raise ValueError(f"No code contract registered for task {task_type}")
    return TASK_CODE_CONTRACTS[task_type]


def executable_api_rules(task_type: str) -> str:
    """The host protocol takes precedence over descriptive business output names."""
    contract = get_task_contract(task_type)
    return (
        "HOST EXECUTION CONTRACT (mandatory): Write module-level functions, never a class or methods. "
        "No self arguments. Define exactly: def train(train_df, target_col, config=None):; "
        "def predict(model, test_df):; def evaluate(model, test_df, target_col):; def metadata():. "
        "train returns ONE fitted sklearn Pipeline containing preprocessing and the estimator. "
        "predict receives FEATURES ONLY and calls this fitted Pipeline directly; never drop or access the target in predict. "
        f"predict returns a pandas DataFrame with these EXACT column names: {list(contract.required_outputs)}. "
        "Business output_schema labels describe meaning, not Python column names. "
        "evaluate receives a labeled DataFrame: split its target_col from features and compute real metrics. "
        "metadata returns a dictionary with algorithm, rationale, evidence_ids. "
        "Include ALL necessary imports. ColumnTransformer is from sklearn.compose. Keep code concise, with short comments. "
        "Avoid feature-name introspection inside predict: let the fitted Pipeline validate the feature frame. "
        "The prediction column contains discrete class labels for classification, never probabilities. "
        "Binary F1/precision/recall use average='binary'; text classification uses average='weighted'. "
        "If reporting pr_auc, use sklearn.metrics.average_precision_score(y_true, positive_class_probability), matching the trusted evaluator's definition. "
        + ("For text, construct Pipeline([text selection FunctionTransformer, TfidfVectorizer, classifier]). "
           "The text selection step must return a ONE-dimensional Series of strings via frame[text_column].fillna('').astype(str). "
           "TfidfVectorizer consumes 1D strings and produces a 2D sparse feature matrix. "
           "Do not wrap raw text selection in ColumnTransformer, reshape strings to (-1,1), or call .values on text to make 2D arrays. "
           "Config is a mixed search-plan dictionary, not constructor kwargs: explicitly route max_features to TfidfVectorizer; "
           "translate ngram_max into ngram_range=(1, ngram_max); route C, max_iter, class_weight and random_state only to LogisticRegression. "
           "Never pass **config to either component. Ignore orchestration keys such as scaler, threshold and ngram_max in estimator constructors. "
           if task_type == "text_classification" else "Tabular preprocessing must impute missing numeric and categorical values; OneHotEncoder must use handle_unknown='ignore'. ")
        + ("Also define module-level def predict_proba(model, test_df): returning model.predict_proba(test_df)[:, 1]. " if task_type == "binary_classification" else "")
        + "Implement every function fully. Do not create CoderAgent, RepairAgent or Algorithm classes."
    )


def build_codegen_prompt(spec: Any, plan: Any) -> str:
    contract = get_task_contract(spec.task_type)
    outputs = ", ".join(contract.required_outputs + contract.optional_outputs)
    metrics = ", ".join(contract.metrics)
    return (
        f"Task type: {contract.task_type}. Required prediction outputs: {outputs}. Metrics: {metrics}. "
        f"Target column is passed dynamically as target_col='{spec.target_column}'. Features: {spec.feature_columns}. "
        f"Algorithm: {plan.algorithm_name}; parameters: {plan.hyperparameters}. "
        "Exact API: train(train_df, target_col, config=None), predict(model, test_df), evaluate(model, test_df, target_col), metadata(). "
        "train must return one object encapsulating preprocessing and estimator, e.g. an sklearn Pipeline. Never add a preprocessor parameter to predict/evaluate. "
        "metadata() returns a dict including algorithm, rationale and evidence_ids. predict returns a pandas DataFrame with the requested columns. "
        "For classification with probability output, also implement predict_proba(model,test_df) returning a 1D positive-class probability vector; predict probability column must match it. "
        "Use config=config or {} and honor config.get('random_state',42) for estimator seeds. For binary classification F1 uses binary average; text classification uses weighted F1. "
        + ("For text, use a Pipeline with FunctionTransformer selecting the text column and fillna('').astype(str), TfidfVectorizer, and the classifier. Do not one-hot encode free text. " if spec.task_type == "text_classification" else "Fit preprocessing only on training features, with numeric imputation and categorical imputation + OneHotEncoder(handle_unknown='ignore', sparse_output=False). ")
        +
        "predict must handle missing values, unseen categories and a one-row batch. Empty/invalid input may raise ValueError. No target column is supplied to predict. "
        + " ".join(contract.prompt_requirements)
        + " Never hard-code a target column in predict/evaluate. Never pass prediction threshold to an estimator constructor. "
        "Only use pandas, numpy and scikit-learn; no files, network or system calls.\n"
        + json.dumps({"requirement": spec.to_dict(), "execution_plan": plan.to_dict()}, ensure_ascii=False)
        + "\n" + executable_api_rules(spec.task_type)
    )
