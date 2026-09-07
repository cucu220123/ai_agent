"""Subprocess execution with sanitized environment, hard timeout and bounded resources."""
from __future__ import annotations
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def network_namespace_available() -> bool:
    binary = shutil.which("unshare")
    if os.name != "posix" or not binary:
        return False
    try:
        return subprocess.run([binary, "-Urn", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_isolated(algorithm_path: str | Path, data_path: str | Path, target_col: str, timeout_seconds: int, memory_mb: int = 8192, expected_outputs: list[str] | None = None, task_type: str = "binary_classification", *, config: dict | None = None, repeats: int = 3, cv_folds: int = 0, require_probability: bool = False) -> dict[str, Any]:
    # Never inherit the API credential, user HOME, PYTHONPATH or proxy configuration.
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH") if key in os.environ}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    with tempfile.TemporaryDirectory(prefix="ai_factory_sandbox_") as folder:
        root = Path(folder)
        shutil.copyfile(algorithm_path, root / "algorithm.py")
        shutil.copyfile(data_path, root / "dataset.csv")
        output = root / (uuid.uuid4().hex + ".json")
        request = {"algorithm_path": str(root / "algorithm.py"), "data_path": str(root / "dataset.csv"), "target": target_col, "task_type": task_type, "outputs": expected_outputs or ["prediction"], "timeout_seconds": timeout_seconds, "memory_mb": memory_mb, "config": config or {}, "repeats": max(2, repeats), "cv_folds": cv_folds, "require_probability": require_probability, "result_path": str(output)}
        request_path = root / "request.json"
        request_path.write_text(json.dumps(request))
        worker = Path(__file__).with_name("worker.py").resolve()
        command = [sys.executable, "-I", str(worker), str(request_path)]
        namespaced = network_namespace_available()
        if namespaced:
            command = [shutil.which("unshare"), "-Urn", "--", *command]
        with (root / "stdout.log").open("w") as stdout_file, (root / "stderr.log").open("w") as stderr_file:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout_file, stderr=stderr_file, start_new_session=os.name == "posix")
            timeout = False
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timeout = True
                if os.name == "posix": os.killpg(process.pid, signal.SIGKILL)
                else: process.kill()
                process.wait()
        stdout = (root / "stdout.log").read_text(errors="replace")[-10000:]
        stderr = (root / "stderr.log").read_text(errors="replace")[-10000:]
        if timeout or process.returncode or not output.exists():
            return {"passed": False, "timeout": timeout, "returncode": process.returncode, "message": f"isolated validation timeout after {timeout_seconds}s" if timeout else f"isolated process exit={process.returncode}", "stdout": stdout, "stderr": stderr, "environment_sanitized": True}
        result = json.loads(output.read_text())
        result["sandbox"]["network_namespace"] = namespaced
        # Compatibility metrics are independently recomputed here in the parent.
        import pandas as pd
        from app.validation.evaluation import split_frames, trusted_metrics
        _, test = split_frames(pd.read_csv(data_path), target_col, task_type)
        metrics = trusted_metrics(test, result["records"][0]["prediction"], target_col, task_type)
        return {"passed": True, "timeout": False, "message": "isolated execution passed", "stdout": stdout, "stderr": stderr, "metrics": metrics, "environment_sanitized": True, "resource_limits_applied": result["resource_limits"]["applied"], **result}

