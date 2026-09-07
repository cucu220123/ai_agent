"""Unified validation: safe execution, independently scored outputs and explicit budgets."""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from app.models import CapabilitySpec, ValidationResult
from app.validation.checks import static_check, import_check
from app.validation.isolate import run_isolated
from app.validation.dataset import profile_dataset
from app.validation.evaluation import split_frames, trusted_metrics
from app.validation.semantic import GeneratedCodeSemanticValidator
from app.metrics.registry import METRIC_REGISTRY


class ValidationRunner:
    def __init__(self, timeout_seconds: int = 90, memory_mb: int = 8192, seed_variance: float = .02, repeats: int = 3, cv_folds: int = 0):
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb
        self.seed_variance = seed_variance
        self.repeats = repeats
        self.cv_folds = cv_folds

    def run(self, algorithm_path: str | Path, data_path: str | Path, spec: CapabilitySpec, algorithm_name: str, repair_round: int = 0, config: dict | None = None) -> ValidationResult:
        started = time.perf_counter()
        result = ValidationResult(status="failed", algorithm=algorithm_name, task_type=spec.task_type, repair_round=repair_round)
        checks, errors = result.checks, result.errors
        try:
            safety = static_check(algorithm_path)
            checks["static_safety"] = safety
            if not safety["passed"]:
                result.failure_type = "syntax_failure" if "syntax error" in safety["message"] else "interface_failure" if safety.get("category") == "interface" else "security_failure"
                raise ValueError(safety["message"])
            checks["interface_import"] = import_check(algorithm_path)
            semantic = GeneratedCodeSemanticValidator().validate_path(str(algorithm_path), spec.task_type, spec.target_column)
            checks["semantic_contract"] = semantic.to_dict()
            if not semantic.passed:
                result.failure_type = "interface_failure"
                raise ValueError(semantic.to_dict()["message"])
            frame = pd.read_csv(data_path)
            result.dataset_profile = profile_dataset(data_path, spec.target_column)
            if len(frame) < 8:
                raise ValueError("validation requires at least 8 rows")
            if spec.target_column:
                if spec.target_column not in frame:
                    raise ValueError("target column not found")
                if frame[spec.target_column].isna().any():
                    raise ValueError("target contains missing labels")
                if spec.task_type in {"binary_classification", "text_classification", "multiclass_classification"} and frame[spec.target_column].nunique() < 2:
                    raise ValueError("classification requires at least two classes")
            duplicates = [c for c in frame if c != spec.target_column and spec.target_column and frame[c].equals(frame[spec.target_column])]
            checks["target_leakage"] = {"passed": not duplicates, "duplicate_target_columns": duplicates, "scope": "exact duplicate label detection; semantic leakage needs domain review"}
            if duplicates:
                raise ValueError(f"target leakage detected: {duplicates}")
            _, test = split_frames(frame, spec.target_column, spec.task_type)
            timeout = min(self.timeout_seconds, int(spec.resource_constraints.get("max_runtime_seconds", self.timeout_seconds)))
            memory = min(self.memory_mb, int(spec.resource_constraints.get("max_memory_mb", self.memory_mb)))
            probability_required = spec.probability_output_required and spec.task_type in {"binary_classification", "text_classification", "multiclass_classification"}
            outputs = list(spec.output_schema) or spec.output_columns or ["prediction"]
            if probability_required and "probability" not in outputs:
                outputs.append("probability")
            isolated = run_isolated(algorithm_path, data_path, spec.target_column, max(1, timeout), max(64, memory), outputs, spec.task_type, config=config, repeats=self.repeats, cv_folds=self.cv_folds, require_probability=probability_required)
            result.stdout, result.stderr = isolated.get("stdout", ""), isolated.get("stderr", "")
            checks["isolated_execution"] = {k: v for k, v in isolated.items() if k not in {"stdout", "stderr", "records", "cv_records", "metrics", "metadata"}}
            if not isolated["passed"]:
                lower = result.stderr.lower()
                result.failure_type = "resource_limit" if isolated.get("timeout") or isolated.get("returncode") in {-9, -24, -25} or any(t in lower for t in ("memoryerror", "cannot allocate", "memory allocation")) else "runtime_failure"
                raise RuntimeError(isolated["message"] + ": " + result.stderr[-2500:])
            measured = [trusted_metrics(test, record["prediction"], spec.target_column, spec.task_type) for record in isolated["records"]]
            result.metrics = measured[0]
            if spec.target_column:
                for record, metrics in zip(isolated["records"], measured):
                    metrics.update(METRIC_REGISTRY.compute_custom(spec.task_type, test[spec.target_column].to_numpy(), record["prediction"]))
            disagreements = {}
            reported = isolated["records"][0]["reported_metrics"]
            for metric, actual in measured[0].items():
                if metric in reported and not math.isclose(actual, reported[metric], rel_tol=1e-6, abs_tol=1e-8):
                    disagreements[metric] = {"trusted": actual, "generated": reported[metric]}
            checks["metric_integrity"] = {"passed": not disagreements, "source": "parent_process_trusted_metrics", "disagreements": disagreements}
            if disagreements:
                result.failure_type = "metric_integrity_failure"
                errors.append("generated evaluate metrics disagree with independently recomputed predictions")
            metric_pass = {metric: METRIC_REGISTRY.passes(metric, result.metrics.get(metric), threshold) for metric, threshold in spec.metric_thresholds.items()}
            checks["metrics"] = {"passed": all(metric_pass.values()), "per_metric": metric_pass, "thresholds": spec.metric_thresholds, "actual": result.metrics}
            if not all(metric_pass.values()):
                result.failure_type = result.failure_type or "metric_underperformance"
                errors.extend(f"{metric}={result.metrics.get(metric)} violates threshold {spec.metric_thresholds[metric]}" for metric, passed in metric_pass.items() if not passed)
            drift = max((abs(measured[0][metric] - measured[1][metric]) for metric in measured[0]), default=0)
            variance = {metric: float(np.var([item[metric] for item in measured])) for metric in measured[0]}
            normalized_variance = {metric: value / max(1.0, abs(float(np.mean([item[metric] for item in measured]))) ** 2) for metric, value in variance.items()}
            stable = drift <= 1e-9 and max(normalized_variance.values(), default=0) <= self.seed_variance
            checks["stability"] = {"passed": stable, "same_seed_drift": drift, "seeds": [r["seed"] for r in isolated["records"]], "metric_variance": variance, "normalized_variance": normalized_variance, "variance_limit": self.seed_variance}
            if not stable:
                result.failure_type = result.failure_type or "instability"
                errors.append(f"stability failure: drift={drift}, normalized variance={normalized_variance}")
            latency = isolated["records"][0]["latency_ms_per_row"]
            checks["latency"] = {"passed": spec.latency_requirement_ms is None or latency <= spec.latency_requirement_ms, "ms_per_row": latency, "limit_ms_per_row": spec.latency_requirement_ms}
            if not checks["latency"]["passed"]:
                result.failure_type = result.failure_type or "resource_limit"
                errors.append("prediction latency exceeds requirement")
            result.resource_usage = {"max_rss_kb": isolated.get("max_rss_kb"), "cpu_seconds": isolated.get("cpu_seconds"), "latency_ms_per_row": latency, "limits": isolated["resource_limits"], "sandbox": isolated["sandbox"]}
            checks["resource_usage"] = {"passed": not isolated.get("max_rss_kb") or isolated["max_rss_kb"] <= memory * 1024, **result.resource_usage}
            if not checks["resource_usage"]["passed"]:
                result.failure_type = result.failure_type or "resource_limit"
                errors.append("peak RSS exceeds memory budget")
            checks["functional"] = {"passed": True, "prediction_rows": isolated["prediction_rows"]}
            checks["output_contract"] = {"passed": True, "required_outputs": outputs, "metadata": isolated.get("metadata", {})}
            checks["robustness"] = {"passed": True, **isolated["robustness"]}
            cv = [trusted_metrics(frame.iloc[item["indices"]], item["prediction"], spec.target_column, spec.task_type) for item in isolated["cv_records"]]
            if cv:
                checks["cross_validation"] = {"passed": True, "folds": cv, "mean": {k: float(np.mean([v[k] for v in cv])) for k in cv[0]}}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            result.failure_type = result.failure_type or "runtime_failure"
            checks.setdefault("functional", {"passed": False, "message": str(exc)[:1000]})
        result.runtime_seconds = time.perf_counter() - started
        checks["runtime_budget"] = {"passed": result.runtime_seconds <= self.timeout_seconds + 2, "runtime_seconds": result.runtime_seconds, "timeout_seconds": self.timeout_seconds}
        result.status = "passed" if not errors else "failed"
        result.root_cause = errors[0] if errors else None
        return result

