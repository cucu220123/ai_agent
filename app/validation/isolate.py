from __future__ import annotations

import json
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
from sklearn.model_selection import train_test_split

algorithm_path, data_path, target_col = sys.argv[1:4]
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
model = module.train(train_df, target_col, {"random_state": 42, "threshold": 0.5})
prediction_frame = test_df.drop(columns=[target_col]) if target_col and target_col in test_df.columns else test_df
predictions = module.predict(model, prediction_frame)
metrics = {str(k): float(v) for k, v in module.evaluate(model, test_df, target_col).items()}
if len(predictions) != len(test_df):
    raise ValueError("prediction row count does not match test rows")
if "prediction" not in predictions.columns:
    raise ValueError("prediction output must contain prediction")
if "probability" in predictions.columns and (predictions["probability"].isna().any() or ((predictions["probability"] < 0) | (predictions["probability"] > 1)).any()):
    raise ValueError("probability must be finite and in [0, 1]")
model2 = module.train(train_df, target_col, {"random_state": 42, "threshold": 0.5})
metrics2 = {str(k): float(v) for k, v in module.evaluate(model2, test_df, target_col).items()}
drift = max((abs(metrics.get(k, 0.0) - metrics2.get(k, 0.0)) for k in metrics), default=0.0)
positive_rate = float(test_df[target_col].mean()) if has_target and pd.api.types.is_numeric_dtype(test_df[target_col]) else 0.0
print("AI_FACTORY_RESULT=" + json.dumps({"metrics": metrics, "metrics2": metrics2, "drift": drift, "prediction_rows": len(predictions), "positive_rate": positive_rate, "runtime_seconds": time.perf_counter() - started, "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
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


def run_isolated(algorithm_path: str | Path, data_path: str | Path, target_col: str, timeout_seconds: int, memory_mb: int = 16384) -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory(prefix="ai_factory_sandbox_") as temp_dir:
            command = [sys.executable, "-I", "-c", HARNESS, str(Path(algorithm_path).resolve()), str(Path(data_path).resolve()), target_col]
            limits_applied = True
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False, cwd=temp_dir, preexec_fn=lambda: _limit_resources(timeout_seconds, memory_mb))
            except subprocess.SubprocessError:
                # Some managed/server environments reject preexec resource limits. Keep hard timeout and isolated cwd.
                limits_applied = False
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
