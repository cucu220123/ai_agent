from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


HARNESS = r'''
import importlib.util
import json
import sys
import time
import resource
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

algorithm_path, data_path, target_col, expected_outputs_json, task_type = sys.argv[1:6]
expected_outputs = set(json.loads(expected_outputs_json))
started = time.perf_counter()
spec = importlib.util.spec_from_file_location("isolated_generated_algorithm", algorithm_path)
if spec is None or spec.loader is None:
    raise ImportError("cannot create module spec")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
df = pd.read_csv(data_path)
has_target = bool(target_col) and target_col in df.columns
if has_target and df[target_col].nunique() >= 2:
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[target_col] if df[target_col].nunique() < 10 else None)
else:
    train_df, test_df = df, df
model = module.train(train_df, target_col, {"random_state": 42})
prediction_frame = test_df.drop(columns=[target_col]) if target_col and target_col in test_df.columns else test_df
predictions = module.predict(model, prediction_frame)
metrics = {str(k): float(v) for k, v in module.evaluate(model, test_df, target_col).items()}
if len(predictions) != len(test_df):
    raise ValueError("prediction row count does not match test rows")
missing_outputs = expected_outputs - set(predictions.columns)
if missing_outputs:
    raise ValueError(f"prediction output missing {sorted(missing_outputs)}")
if "probability" in predictions.columns and (predictions["probability"].isna().any() or ((predictions["probability"] < 0) | (predictions["probability"] > 1)).any()):
    raise ValueError("probability must be finite and in [0, 1]")
numeric_output = predictions.select_dtypes(include=["number"])
if numeric_output.isna().any().any() or (not np.isfinite(numeric_output.to_numpy(dtype=float)).all()):
    raise ValueError("numeric predictions must not contain NaN or infinity")
small_batch = module.predict(model, prediction_frame.iloc[:1].copy())
if len(small_batch) != 1:
    raise ValueError("small batch prediction must preserve row count")
robust_frame = prediction_frame.iloc[: min(5, len(prediction_frame))].copy()
for column in robust_frame.columns:
    if pd.api.types.is_numeric_dtype(robust_frame[column]):
        robust_frame.loc[robust_frame.index[0], column] = float("nan")
    elif len(robust_frame):
        robust_frame.loc[robust_frame.index[0], column] = "__UNSEEN_CATEGORY__"
robust_predictions = module.predict(model, robust_frame)
if len(robust_predictions) != len(robust_frame):
    raise ValueError("robustness prediction row count mismatch")
empty_behavior = "accepted"
try:
    empty_predictions = module.predict(model, prediction_frame.iloc[:0].copy())
    if len(empty_predictions) != 0:
        raise ValueError("empty input returned non-empty predictions")
except Exception:
    empty_behavior = "rejected"
model2 = module.train(train_df, target_col, {"random_state": 42})
metrics2 = {str(k): float(v) for k, v in module.evaluate(model2, test_df, target_col).items()}
model3 = module.train(train_df, target_col, {"random_state": 7})
metrics3 = {str(k): float(v) for k, v in module.evaluate(model3, test_df, target_col).items()}
drift = max((abs(metrics.get(k, 0.0) - metrics2.get(k, 0.0)) for k in metrics), default=0.0)
variance = {key: float(pd.Series([metrics.get(key), metrics2.get(key), metrics3.get(key)]).var(ddof=0)) for key in metrics}
positive_rate = float(test_df[target_col].mean()) if has_target and pd.api.types.is_numeric_dtype(test_df[target_col]) else 0.0
usage = resource.getrusage(resource.RUSAGE_SELF)
print("AI_FACTORY_RESULT=" + json.dumps({"metrics": metrics, "metrics2": metrics2, "metrics3": metrics3, "metric_variance": variance, "drift": drift, "prediction_rows": len(predictions), "positive_rate": positive_rate, "robustness": {"small_batch": True, "missing_values_and_unseen_categories": True, "empty_input": empty_behavior}, "runtime_seconds": time.perf_counter() - started, "cpu_seconds": usage.ru_utime + usage.ru_stime, "max_rss_kb": usage.ru_maxrss}))
'''


def _limit_resources(cpu_seconds: int, memory_mb: int) -> None:
    try:
        if cpu_seconds > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        if memory_mb > 0:
            memory_bytes = memory_mb * 1024 * 1024
            # Do not lower a process soft limit if the host already configured a smaller hard limit.
            hard = resource.getrlimit(resource.RLIMIT_AS)[1]
            if hard < 0 or memory_bytes <= hard:
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, hard if hard >= 0 else memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    except (ValueError, OSError):
        pass


def run_isolated(algorithm_path: str | Path, data_path: str | Path, target_col: str, timeout_seconds: int, memory_mb: int = 16384, expected_outputs: list[str] | None = None, task_type: str = "binary_classification") -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory(prefix="ai_factory_sandbox_") as temp_dir:
            command = [sys.executable, "-I", "-c", HARNESS, str(Path(algorithm_path).resolve()), str(Path(data_path).resolve()), target_col, json.dumps(expected_outputs or ["prediction"]), task_type]
            limits_applied = os.getenv("SANDBOX_ENABLE_RLIMITS", "0") == "1"
            if limits_applied:
                try:
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False, cwd=temp_dir, preexec_fn=lambda: _limit_resources(timeout_seconds, memory_mb))
                except subprocess.SubprocessError:
                    limits_applied = False
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False, cwd=temp_dir)
            else:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False, cwd=temp_dir)
    except subprocess.TimeoutExpired as exc:
        return {"passed": False, "timeout": True, "message": f"isolated validation exceeded {timeout_seconds}s", "stdout": (exc.stdout or "")[-5000:], "stderr": (exc.stderr or "")[-5000:]}
    stdout, stderr = completed.stdout[-10000:], completed.stderr[-10000:]
    marker = next((line for line in stdout.splitlines() if line.startswith("AI_FACTORY_RESULT=")), None)
    if completed.returncode != 0 or marker is None:
        return {"passed": False, "timeout": False, "message": f"isolated process exit={completed.returncode}", "stdout": stdout, "stderr": stderr}
    try:
        result = json.loads(marker.split("=", 1)[1])
    except json.JSONDecodeError as exc:
        return {"passed": False, "timeout": False, "message": f"invalid harness output: {exc}", "stdout": stdout, "stderr": stderr}
    return {"passed": True, "timeout": False, "message": "isolated execution passed", "resource_limits_applied": locals().get("limits_applied", False), "stdout": stdout, "stderr": stderr, **result}
