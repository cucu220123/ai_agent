"""Trusted subprocess harness. Do not import generated code in the service process."""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# -I intentionally removes cwd from imports; this is the trusted package root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def set_limits(cpu: int, memory_mb: int) -> dict:
    if os.name != "posix":
        return {"applied": False, "reason": "resource module unavailable on this OS"}
    import resource
    for kind, limit in ((resource.RLIMIT_CPU, max(1, cpu)), (resource.RLIMIT_AS, memory_mb * 1024 * 1024), (resource.RLIMIT_FSIZE, 16 * 1024 * 1024)):
        if limit > 0:
            _, hard = resource.getrlimit(kind)
            value = min(limit, hard) if hard >= 0 else limit
            resource.setrlimit(kind, (value, value))
    return {"applied": True, "cpu_seconds": cpu, "address_space_mb": memory_mb, "max_file_mb": 16}


def install_audit_guard(sandbox: Path) -> dict:
    roots = [sandbox.resolve(), Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    if os.name == "posix":
        roots.extend([Path("/usr/lib"), Path("/lib"), Path("/usr/share/zoneinfo")])

    def audit(event, args):
        if event.startswith(("socket.", "subprocess.", "os.exec", "os.spawn")) or event in {"os.system", "os.fork", "pty.spawn"}:
            raise PermissionError("prototype sandbox blocks network and process creation")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode = args[1] if len(args) > 1 else "r"
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and sandbox not in path.parents:
                raise PermissionError("prototype sandbox write outside temporary directory")
            if not writing and path != Path(os.devnull) and not any(path == root or root in path.parents for root in roots):
                raise PermissionError("prototype sandbox read outside whitelisted roots")
    sys.addaudithook(audit)
    return {"network_audit_blocked": True, "filesystem_audit_whitelist": True, "production_isolation": False}


def main() -> None:
    started = time.perf_counter()
    request = json.loads(Path(sys.argv[1]).read_text())
    limits = set_limits(request["timeout_seconds"], request["memory_mb"])
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, KFold
    from app.validation.evaluation import evaluation_frames, validate_outputs
    frame = pd.read_csv(request["data_path"])
    target, task = request["target"], request["task_type"]
    final_frame = pd.read_csv(request["evaluation_data_path"]) if request.get("evaluation_data_path") else None
    train, test = evaluation_frames(frame, target, task, final_frame)
    features = test.drop(columns=[target]) if target else test.copy()
    guard = install_audit_guard(Path.cwd().resolve())
    module_spec = importlib.util.spec_from_file_location("generated_algorithm", request["algorithm_path"])
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    outputs = request["outputs"]
    records = []
    trained = None
    for index in range(request["repeats"]):
        seed = 42 if index < 2 else 7 + index
        trained = module.train(train.copy(deep=True), target, {**request["config"], "random_state": seed})
        tick = time.perf_counter()
        predicted = validate_outputs(module.predict(trained, features.copy(deep=True)), len(features), outputs)
        latency = (time.perf_counter() - tick) * 1000 / max(1, len(features))
        reported = module.evaluate(trained, test.copy(deep=True), target)
        if not isinstance(reported, dict) or not all(isinstance(v, (int, float, np.number)) and np.isfinite(float(v)) for v in reported.values()):
            raise ValueError("evaluate must return a finite numeric metric dictionary")
        records.append({"seed": seed, "prediction": predicted, "reported_metrics": {k: float(v) for k, v in reported.items()}, "latency_ms_per_row": latency})
    metadata = module.metadata() if hasattr(module, "metadata") else {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must return a dictionary")
    if request["require_probability"]:
        if not callable(getattr(module, "predict_proba", None)):
            raise ValueError("missing required predict_proba interface")
        probability = np.asarray(module.predict_proba(trained, features.copy()))
        if probability.shape != (len(features),) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("predict_proba must return a finite positive-class probability vector")
        expected = np.asarray(module.predict(trained, features.copy())["probability"], dtype=float)
        if not np.allclose(probability, expected, atol=1e-8):
            raise ValueError("predict_proba disagrees with predict probability column")
    validate_outputs(module.predict(trained, features.iloc[:1].copy()), 1, outputs)
    robust = features.iloc[: min(5, len(features))].copy()
    for column in robust.columns:
        if pd.api.types.is_numeric_dtype(robust[column]):
            robust[column] = robust[column].astype(float)
            robust.loc[robust.index[0], column] = np.nan
        else:
            robust.loc[robust.index[0], column] = "__UNSEEN_CATEGORY__"
    validate_outputs(module.predict(trained, robust), len(robust), outputs)
    try:
        empty = module.predict(trained, features.iloc[:0].copy())
    except (ValueError, TypeError, KeyError):
        empty_behavior = "explicitly_rejected"
    else:
        validate_outputs(empty, 0, outputs)
        empty_behavior = "valid_empty_frame"
    try:
        module.predict(trained, None)
    except (ValueError, TypeError, KeyError, AttributeError):
        invalid_behavior = "rejected"
    else:
        raise ValueError("invalid input must be rejected")
    cv_records = []
    folds = request["cv_folds"]
    if folds >= 2 and target:
        classifier = task in {"binary_classification", "text_classification", "multiclass_classification"}
        splitter = StratifiedKFold(folds, shuffle=True, random_state=17) if classifier else KFold(folds, shuffle=True, random_state=17)
        for fit_indices, eval_indices in splitter.split(frame, frame[target] if classifier else None):
            model_cv = module.train(frame.iloc[fit_indices].copy(), target, {**request["config"], "random_state": 42})
            cv_prediction = module.predict(model_cv, frame.iloc[eval_indices].drop(columns=[target]).copy())
            cv_records.append({"indices": eval_indices.tolist(), "prediction": validate_outputs(cv_prediction, len(eval_indices), outputs)})
    resource_usage = {}
    if os.name == "posix":
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        resource_usage = {"max_rss_kb": usage.ru_maxrss, "cpu_seconds": usage.ru_utime + usage.ru_stime}
    result = {"records": records, "cv_records": cv_records, "metadata": metadata, "runtime_seconds": time.perf_counter() - started, "prediction_rows": len(test), "resource_limits": limits, "sandbox": guard, "robustness": {"small_batch": True, "missing_values_and_unseen_categories": True, "empty_input": empty_behavior, "invalid_input": invalid_behavior}, **resource_usage}
    Path(request["result_path"]).write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

