from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.models import CapabilitySpec, ValidationResult
from app.validation.checks import import_check, static_check


class ValidationRunner:
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
        try:
            df = pd.read_csv(data_path)
            if spec.target_column not in df.columns:
                raise ValueError(f"target column '{spec.target_column}' not found in dataset")
            if df[spec.target_column].nunique() < 2:
                raise ValueError("target must contain at least two classes")
            train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[spec.target_column])
            module = imported["module"]
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
                model = module.train(train_df, spec.target_column, {"random_state": 42})
                predictions = module.predict(model, test_df.drop(columns=[spec.target_column]))
                metrics = module.evaluate(model, test_df, spec.target_column)
            stdout, stderr = out.getvalue()[-5000:], err.getvalue()[-5000:]
            checks["functional"] = {"passed": True, "message": "train/predict/evaluate completed", "prediction_rows": int(len(predictions))}
            required = {"prediction", "probability"}
            missing_outputs = required - set(predictions.columns)
            if len(predictions) != len(test_df):
                raise ValueError("prediction row count does not match test rows")
            if missing_outputs:
                raise ValueError(f"missing output columns: {sorted(missing_outputs)}")
            if predictions["probability"].isna().any() or ((predictions["probability"] < 0) | (predictions["probability"] > 1)).any():
                raise ValueError("probability must be finite and in [0, 1]")
            checks["output_contract"] = {"passed": True, "message": "output schema and probability range passed"}
            metrics = {k: float(v) for k, v in metrics.items()}
            checks["metrics"] = {"passed": all(metrics.get(k, 0.0) >= threshold for k, threshold in spec.metric_thresholds.items()), "thresholds": spec.metric_thresholds, "actual": metrics}
            failed_metrics = [f"{k}={metrics.get(k, 0.0):.4f} < {threshold:.4f}" for k, threshold in spec.metric_thresholds.items() if metrics.get(k, 0.0) < threshold]
            if failed_metrics:
                errors.extend(failed_metrics)
            # A second deterministic run is a lightweight stability check.
            model2 = module.train(train_df, spec.target_column, {"random_state": 42})
            metrics2 = module.evaluate(model2, test_df, spec.target_column)
            drift = max(abs(float(metrics.get(k, 0.0)) - float(metrics2.get(k, 0.0))) for k in metrics)
            checks["stability"] = {"passed": drift <= 1e-9, "max_metric_drift": drift}
            if drift > 1e-9:
                errors.append(f"non-deterministic metric drift={drift}")
        except Exception as exc:
            errors.append(f"runtime error: {type(exc).__name__}: {exc}")
            checks.setdefault("functional", {"passed": False, "message": str(exc)})
        runtime = time.perf_counter() - started
        if runtime > self.timeout_seconds:
            errors.append(f"validation runtime {runtime:.2f}s exceeded timeout {self.timeout_seconds}s")
            checks["runtime_budget"] = {"passed": False, "message": errors[-1]}
        else:
            checks["runtime_budget"] = {"passed": True, "message": f"completed within {self.timeout_seconds}s"}
        status = "passed" if not errors else "failed"
        return ValidationResult(status=status, algorithm=algorithm_name, checks=checks, metrics=locals().get("metrics", {}), runtime_seconds=runtime, errors=errors, stdout=stdout, stderr=stderr, repair_round=repair_round)
