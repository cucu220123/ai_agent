from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.models import CapabilitySpec, ValidationResult
from app.validation.checks import import_check, static_check
from app.validation.isolate import run_isolated


class ValidationRunner:
    """Validate generated code in a separate Python process with a hard timeout."""

    def __init__(self, timeout_seconds: int = 90):
        self.timeout_seconds = timeout_seconds

    def run(self, algorithm_path: str | Path, data_path: str | Path, spec: CapabilitySpec, algorithm_name: str, repair_round: int = 0) -> ValidationResult:
        started = time.perf_counter()
        checks: dict[str, dict] = {}
        errors: list[str] = []
        stdout, stderr = "", ""
        static = static_check(algorithm_path)
        checks["static_safety"] = {k: v for k, v in static.items() if k != "module"}
        if not static["passed"]:
            return ValidationResult(status="failed", algorithm=algorithm_name, checks=checks, errors=[static["message"]], runtime_seconds=time.perf_counter() - started, repair_round=repair_round)
        imported = import_check(algorithm_path)
        checks["interface_import"] = {k: v for k, v in imported.items() if k != "module"}
        if not imported["passed"]:
            return ValidationResult(status="failed", algorithm=algorithm_name, checks=checks, errors=[imported["message"]], runtime_seconds=time.perf_counter() - started, repair_round=repair_round)
        metrics: dict[str, float] = {}
        try:
            df = pd.read_csv(data_path)
            if spec.target_column and spec.target_column not in df.columns:
                raise ValueError(f"target column '{spec.target_column}' not found in dataset")
            if spec.task_type == "binary_classification" and df[spec.target_column].nunique() < 2:
                raise ValueError("target must contain at least two classes")
            if spec.target_column:
                stratify = df[spec.target_column] if spec.task_type == "binary_classification" else None
                _, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=stratify)
            else:
                test_df = df
            isolated = run_isolated(algorithm_path, data_path, spec.target_column, self.timeout_seconds)
            stdout, stderr = isolated.get("stdout", ""), isolated.get("stderr", "")
            checks["isolated_execution"] = {k: v for k, v in isolated.items() if k not in {"stdout", "stderr", "metrics", "metrics2"}}
            checks["resource_usage"] = {"max_rss_kb": isolated.get("max_rss_kb"), "runtime_seconds_child": isolated.get("runtime_seconds")}
            if not isolated["passed"]:
                raise RuntimeError(isolated["message"] + (f": {stderr[-1000:]}" if stderr else ""))
            metrics = {k: float(v) for k, v in isolated.get("metrics", {}).items()}
            checks["functional"] = {"passed": True, "message": "isolated train/predict/evaluate completed", "prediction_rows": isolated.get("prediction_rows")}
            checks["output_contract"] = {"passed": True, "message": "output schema and probability range passed"}
            lower_is_better = {"rmse", "mae", "mape", "log_loss"}
            metric_pass = {k: ((metrics.get(k, float("inf")) <= threshold) if k in lower_is_better else (metrics.get(k, 0.0) >= threshold)) for k, threshold in spec.metric_thresholds.items()}
            checks["metrics"] = {"passed": all(metric_pass.values()), "per_metric": metric_pass, "thresholds": spec.metric_thresholds, "actual": metrics}
            failed_metrics = [f"{k}={metrics.get(k, 0.0):.4f} {'>' if k in lower_is_better else '<'} {threshold:.4f}" for k, threshold in spec.metric_thresholds.items() if not metric_pass.get(k, False)]
            if failed_metrics:
                errors.extend(failed_metrics)
            positive_rate = float(isolated.get("positive_rate", test_df[spec.target_column].mean() if spec.target_column else 0.0))
            checks["class_balance"] = {"positive_rate": positive_rate, "train_rows": int(len(df) - len(test_df)), "test_rows": int(len(test_df)), "warning": "positive class is below 10%; consider threshold tuning" if spec.target_column and positive_rate < 0.1 else ("unsupervised task; no target balance check" if not spec.target_column else "class balance acceptable")}
            drift = float(isolated.get("drift", 0.0))
            checks["stability"] = {"passed": drift <= 1e-9, "max_metric_drift": drift}
            if drift > 1e-9:
                errors.append(f"non-deterministic metric drift={drift}")
        except Exception as exc:
            errors.append(f"runtime error: {type(exc).__name__}: {exc}")
            checks.setdefault("functional", {"passed": False, "message": str(exc)})
        runtime = time.perf_counter() - started
        checks["runtime_budget"] = {"passed": runtime <= self.timeout_seconds, "message": f"completed in {runtime:.2f}s (budget {self.timeout_seconds}s)"}
        if runtime > self.timeout_seconds:
            errors.append(f"validation runtime {runtime:.2f}s exceeded timeout {self.timeout_seconds}s")
        status = "passed" if not errors else "failed"
        error_text = " ".join(errors).lower()
        if status == "passed":
            failure_type = None
        elif "runtime" in error_text or "traceback" in error_text or "typeerror" in error_text or "keyerror" in error_text:
            failure_type = "runtime_failure"
        elif any("<" in e or ">" in e for e in errors):
            failure_type = "metric_underperformance"
        else:
            failure_type = "validation_failure"
        return ValidationResult(status=status, algorithm=algorithm_name, checks=checks, metrics=metrics, runtime_seconds=runtime, errors=errors, stdout=stdout, stderr=stderr, repair_round=repair_round, task_type=spec.task_type, dataset_profile={"path": str(data_path), "rows": int(len(df)), "columns": list(df.columns)}, resource_usage={"max_rss_kb": checks.get("resource_usage", {}).get("max_rss_kb")}, failure_type=failure_type)
