from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "errors": self.errors, "message": "; ".join(self.errors) if self.errors else "generated code semantic contract passed"}


class GeneratedCodeSemanticValidator:
    """AST/estimator-aware gate for obvious leakage and task contract mistakes."""

    estimator_classes = {"LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier", "RandomForestRegressor", "IsolationForest"}

    def validate_path(self, path: str, task_type: str, target_column: str = "") -> SemanticValidation:
        return self.validate_source(open(path, encoding="utf-8").read(), task_type, target_column)

    def validate_source(self, source: str, task_type: str = "binary_classification", target_column: str = "") -> SemanticValidation:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return SemanticValidation(False, [f"syntax error: {exc}"])
        errors: list[str] = []
        train_fn = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train"), None)
        predict_fn = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "predict"), None)
        evaluate_fn = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "evaluate"), None)
        if task_type in {"binary_classification", "multiclass_classification", "regression"} and train_fn:
            x_drop_line = None
            for node in ast.walk(train_fn):
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "X":
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "drop":
                        x_drop_line = node.lineno
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "select_dtypes":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "train_df":
                        errors.append("feature discovery must derive from X after target drop, not train_df")
                    if x_drop_line is not None and node.lineno < x_drop_line:
                        errors.append("feature discovery occurs before X=drop(target)")
            for node in ast.walk(train_fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.estimator_classes:
                    try:
                        from sklearn.base import clone
                        import sklearn.linear_model, sklearn.ensemble
                        estimator_type = getattr(sklearn.linear_model, node.func.id, None) or getattr(sklearn.ensemble, node.func.id, None)
                        if estimator_type:
                            allowed = set(estimator_type().get_params())
                            for keyword in node.keywords:
                                if keyword.arg and keyword.arg not in allowed:
                                    errors.append(f"{node.func.id} does not accept estimator kwarg {keyword.arg}")
                    except Exception:
                        pass
            for node in ast.walk(train_fn):
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "config":
                    errors.append("config access must be guarded by config=None normalization and config.get")
        for fn, name in ((predict_fn, "predict"), (evaluate_fn, "evaluate")):
            if not fn:
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "drop":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            errors.append(f"{name} must not hard-code target column {arg.value!r}")
                    for keyword in node.keywords:
                        if isinstance(keyword.value, (ast.List, ast.Tuple)):
                            for item in keyword.value.elts:
                                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                                    errors.append(f"{name} must not hard-code target column {item.value!r}")
        return SemanticValidation(not errors, list(dict.fromkeys(errors)))

