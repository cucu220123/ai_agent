from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.advisor_agent import PlannerAdviceContract
from app.agents.generator_agent import static_check_text
from app.agents.requirement_agent import RequirementContract
from app.generation.templates import render_algorithm
from app.llm.contracts import extract_json_object, extract_python_code
from app.llm.local_transformers import LocalTransformersLLM
from app.models import AlgorithmPlan, CapabilitySpec
from app.validation.runner import ValidationRunner


MODEL_ROOT = Path("/data/public_checkpoints/huggingface_models")
DEFAULT_MODELS = [
    "Qwen2.5-1.5B-Instruct", "Qwen2.5-Coder-3B-Instruct", "Qwen2.5-7B-Instruct",
    "Qwen3-8B", "Qwen2.5-14B-Instruct", "Qwen3-Coder-30B-A3B-Instruct",
]


def scan_models(root: Path = MODEL_ROOT) -> list[dict[str, Any]]:
    candidates = []
    for config_path in sorted(root.glob("*/config.json")):
        name = config_path.parent.name
        if not any(token in name.lower() for token in ("instruct", "chat", "coder", "qwen3-", "embedding")):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        weights = list(config_path.parent.glob("*.safetensors")) + list(config_path.parent.glob("pytorch_model*.bin"))
        weight_bytes = sum(path.stat().st_size for path in weights if path.is_file())
        if not weights or weight_bytes <= 0:
            continue
        candidates.append({
            "name": name, "path": str(config_path.parent), "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []), "context_length": config.get("max_position_embeddings"),
            "dtype": config.get("torch_dtype"), "weight_files": len(weights), "weight_gb": round(weight_bytes / 1e9, 2),
            "roles": classify_roles(name),
        })
    return candidates


def classify_roles(name: str) -> list[str]:
    lowered = name.lower()
    roles = []
    if "embedding" in lowered or "bge" in lowered or "e5" in lowered:
        return ["embedding"]
    if any(token in lowered for token in ("instruct", "chat", "qwen3-")):
        roles.extend(["general_instruction", "structured_json"])
    if "coder" in lowered or "code" in lowered:
        roles.extend(["code_generation", "code_repair"])
    elif roles:
        roles.append("code_generation_secondary")
    return roles


def benchmark_model(model_path: Path) -> dict[str, Any]:
    import torch

    model = LocalTransformersLLM(str(model_path))
    results: dict[str, Any] = {"model": model_path.name, "path": str(model_path), "tasks": {}}
    started_all = time.perf_counter()

    requirement_prompt = json.dumps({
        "user_requirement": "根据客户年龄和登录次数预测客户是否流失，要求 ROC-AUC 不低于 0.75，并输出概率",
        "dataset": {"columns": ["age", "login_count_30d", "churn"], "dtypes": {"age": "int", "login_count_30d": "int", "churn": "int"}},
        "json_schema": RequirementContract.model_json_schema(),
    }, ensure_ascii=False)
    results["tasks"]["requirement_json"] = run_task(model, "requirement", "只输出严格 JSON。", requirement_prompt, lambda raw: RequirementContract.model_validate(extract_json_object(raw)))

    planning_prompt = json.dumps({
        "planning_context": {"requirement": {"task_type": "binary_classification", "metrics": ["roc_auc"]}, "graph_candidates": [{"algorithm_id": "algorithm_logistic_regression", "evidence_node_ids": ["run_1"]}], "similar_historical_runs": [{"run_id": "run_1", "metrics": {"roc_auc": 0.84}}]},
        "allowed_algorithms": ["logistic_regression"], "json_schema": PlannerAdviceContract.model_json_schema(),
    }, ensure_ascii=False)
    results["tasks"]["planner_json"] = run_task(model, "planning", "只输出严格 JSON；evidence_ids 只能用 run_1。", planning_prompt, lambda raw: PlannerAdviceContract.model_validate(extract_json_object(raw)))

    code_prompt = "只输出完整 Python 代码。仅使用 pandas/numpy/scikit-learn；严格定义 train(train_df,target_col,config=None)、predict(model,test_df)、evaluate(model,test_df,target_col)。实现数值和类别特征的 LogisticRegression 二分类，predict 返回 prediction/probability，evaluate 返回 roc_auc/f1。"
    results["tasks"]["code_generation"] = run_task(model, "code_generation", "你是安全代码生成器，只输出代码。", code_prompt, validate_executable_code)

    spec = CapabilitySpec(raw_description="repair benchmark", feature_columns=["age", "login_count_30d"], metric_thresholds={"roc_auc": 0.5})
    plan = AlgorithmPlan("algorithm_logistic_regression", "Logistic Regression", "benchmark", ["imputation"], {"C": 1.0, "max_iter": 500})
    good_code = render_algorithm(spec, plan)
    broken_code = good_code.replace("def predict(model, test_df", "def predict_broken(model, test_df")
    repair_prompt = f"只输出修复后的完整 Python 代码。错误是 missing function: predict。严格接口 train(train_df,target_col,config=None)、predict(model,test_df)、evaluate(model,test_df,target_col)。\n{broken_code}"
    results["tasks"]["traceback_repair"] = run_task(model, "repair", "你是代码修复器，只输出完整代码。", repair_prompt, validate_executable_code)

    results["latency_seconds"] = round(time.perf_counter() - started_all, 3)
    results["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2) if torch.cuda.is_available() else 0.0
    successes = [bool(item.get("success")) for item in results["tasks"].values()]
    results["success_rate"] = sum(successes) / len(successes)
    results["schema_validity"] = sum(bool(results["tasks"][name].get("success")) for name in ("requirement_json", "planner_json")) / 2
    results["executable_code_rate"] = float(bool(results["tasks"]["code_generation"].get("success")))
    results["repair_success_rate"] = float(bool(results["tasks"]["traceback_repair"].get("success")))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_task(model: LocalTransformersLLM, purpose: str, system: str, prompt: str, validator) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        raw = model.complete(system, prompt, purpose=purpose)
        validation = validator(raw)
        return {"success": True, "latency_seconds": round(time.perf_counter() - started, 3), "usage": model.last_usage, "generation": model.last_generation, "output_preview": raw[:500], "validation": str(type(validation).__name__)}
    except Exception as exc:
        return {"success": False, "latency_seconds": round(time.perf_counter() - started, 3), "usage": model.last_usage, "generation": model.last_generation, "error": f"{type(exc).__name__}: {exc}"[:1500]}


def validate_executable_code(raw: str) -> bool:
    source = extract_python_code(raw)
    if not source:
        raise ValueError("no complete Python module extracted")
    gate = static_check_text(source)
    if not gate["passed"]:
        raise ValueError(gate["message"])
    with tempfile.TemporaryDirectory(prefix="model_benchmark_") as temp:
        temp_path = Path(temp)
        algorithm_path = temp_path / "algorithm.py"
        algorithm_path.write_text(source, encoding="utf-8")
        # Small deterministic dataset, sufficient to exercise the contract.
        rows = ["age,login_count_30d,churn"] + [f"{20+i},{i%12},{1 if i%5==0 else 0}" for i in range(80)]
        data_path = temp_path / "data.csv"
        data_path.write_text("\n".join(rows), encoding="utf-8")
        result = ValidationRunner(30).run(algorithm_path, data_path, CapabilitySpec(raw_description="benchmark", feature_columns=["age", "login_count_30d"], metric_thresholds={}), "benchmark")
        if result.status != "passed":
            raise ValueError("sandbox validation failed: " + "; ".join(result.errors)[:1000])
    return True


def write_reports(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Local Model Benchmark", "", "| Model | Weight GB | Overall | JSON schema | Executable code | Repair | Latency s | Peak GPU MB |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    metadata = {item["name"]: item for item in payload["scanned_models"]}
    for result in payload["benchmarks"]:
        info = metadata.get(result["model"], {})
        md.append(f"| {result['model']} | {info.get('weight_gb', 0)} | {result.get('success_rate', 0):.2f} | {result.get('schema_validity', 0):.2f} | {result.get('executable_code_rate', 0):.2f} | {result.get('repair_success_rate', 0):.2f} | {result.get('latency_seconds', 0):.1f} | {result.get('peak_gpu_memory_mb', 0):.0f} |")
    output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output", default="docs/evidence/local_model_benchmark.json")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--worker-model")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    if args.worker_model:
        result = benchmark_model(Path(args.worker_model))
        print("AI_FACTORY_BENCHMARK=" + json.dumps(result, ensure_ascii=False))
        return 0
    scanned = scan_models()
    if args.scan_only:
        print(json.dumps(scanned, ensure_ascii=False, indent=2))
        return 0
    benchmarks = []
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        path = MODEL_ROOT / name
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-model", str(path)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.getenv("BENCHMARK_CUDA_VISIBLE_DEVICES", "0"), "LOCAL_LLM_DEVICE": "cuda:0"}
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, env=env, check=False)
            marker = next((line for line in completed.stdout.splitlines() if line.startswith("AI_FACTORY_BENCHMARK=")), None)
            if marker:
                benchmarks.append(json.loads(marker.split("=", 1)[1]))
            else:
                benchmarks.append({"model": name, "success_rate": 0.0, "load_or_runtime_error": (completed.stderr or completed.stdout)[-2000:], "latency_seconds": round(time.perf_counter() - started, 3)})
        except subprocess.TimeoutExpired:
            benchmarks.append({"model": name, "success_rate": 0.0, "load_or_runtime_error": f"timeout after {args.timeout}s", "latency_seconds": float(args.timeout)})
        print(json.dumps(benchmarks[-1], ensure_ascii=False), flush=True)
    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": os.getenv("BENCHMARK_CUDA_VISIBLE_DEVICES", "0"), "scanned_models": scanned, "benchmarks": benchmarks}
    write_reports(payload, PROJECT_ROOT / args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
