from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HARNESS = r'''
import importlib.util
import json
import sys
import time
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
if target_col not in df.columns:
    raise ValueError(f"target column '{target_col}' not found")
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[target_col])
model = module.train(train_df, target_col, {"random_state": 42, "threshold": 0.5})
predictions = module.predict(model, test_df.drop(columns=[target_col]))
metrics = {str(k): float(v) for k, v in module.evaluate(model, test_df, target_col).items()}
if len(predictions) != len(test_df):
    raise ValueError("prediction row count does not match test rows")
if not {"prediction", "probability"}.issubset(set(predictions.columns)):
    raise ValueError("prediction output must contain prediction and probability")
if predictions["probability"].isna().any() or ((predictions["probability"] < 0) | (predictions["probability"] > 1)).any():
    raise ValueError("probability must be finite and in [0, 1]")
model2 = module.train(train_df, target_col, {"random_state": 42, "threshold": 0.5})
metrics2 = {str(k): float(v) for k, v in module.evaluate(model2, test_df, target_col).items()}
drift = max((abs(metrics.get(k, 0.0) - metrics2.get(k, 0.0)) for k in metrics), default=0.0)
print("AI_FACTORY_RESULT=" + json.dumps({"metrics": metrics, "metrics2": metrics2, "drift": drift, "prediction_rows": len(predictions), "positive_rate": float(test_df[target_col].mean()), "runtime_seconds": time.perf_counter() - started}))
'''


def run_isolated(algorithm_path: str | Path, data_path: str | Path, target_col: str, timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", HARNESS, str(Path(algorithm_path).resolve()), str(Path(data_path).resolve()), target_col],
            capture_output=True, text=True, timeout=timeout_seconds, check=False,
        )
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
    return {"passed": True, "timeout": False, "message": "isolated execution passed", "stdout": stdout, "stderr": stderr, **result}
