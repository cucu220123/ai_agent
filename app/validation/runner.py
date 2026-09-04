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
            if spec.target_column not in df.columns:
                raise ValueError(f"target column '{spec.target_column}' not found in dataset")
            if df[spec.target_column].nunique() < 2:
                raise ValueError("target must contain at least two classes")
            _, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[spec.target_column])
            isolated = run_isolated(algorithm_path, data_path, spec.target_column, self.timeout_seconds)
            stdout, stderr = isolated.get("stdout", ""), isolated.get("stderr", "")
            checks["isolated_execution"] = {k: v for k, v in isolated.items() if k not in {"stdout", "stderr", "metrics", "metrics2"}}
            if not isolated["passed"]:
                raise RuntimeError(isolated["message"] + (f": {stderr[-1000:]}" if stderr else ""))
            metrics = {k: float(v) for k, v in isolated.get("metrics", {}).items()}
            checks["functional"] = {"passed": True, "message": "isolated train/predict/evaluate completed", "prediction_rows": isolated.get("prediction_rows")}
            checks["output_contract"] = {"passed": True, "message": "output schema and probability range passed"}
            checks["metrics"] = {"passed": all(metrics.get(k, 0.0) >= threshold for k, threshold in spec.metric_thresholds.items()), "thresholds": spec.metric_thresholds, "actual": metrics}
            failed_metrics = [f"{k}={metrics.get(k, 0.0):.4f} < {threshold:.4f}" for k, threshold in spec.metric_thresholds.items() if metrics.get(k, 0.0) < threshold]
            if failed_metrics:
                errors.extend(failed_metrics)
            positive_rate = float(isolated.get("positive_rate", test_df[spec.target_column].mean()))
            checks["class_balance"] = {"positive_rate": positive_rate, "train_rows": int(len(df) - len(test_df)), "test_rows": int(len(test_df)), "warning": "positive class is below 10%; consider threshold tuning" if positive_rate < 0.1 else "class balance acceptable"}
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
        return ValidationResult(status=status, algorithm=algorithm_name, checks=checks, metrics=metrics, runtime_seconds=runtime, errors=errors, stdout=stdout, stderr=stderr, repair_round=repair_round)
