from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.critic_agent import CriticAgent
from app.agents.generator_agent import GeneratorAgent
from app.agents.repair_agent import RepairAgent
from app.experience.retriever import ExperienceRetriever
from app.llm.local_transformers import LocalTransformersLLM
from app.models import AlgorithmPlan, CapabilitySpec
from app.validation.runner import ValidationRunner
from scripts.generate_demo_data import generate


def main() -> int:
    output_dir = PROJECT_ROOT / "examples/real_llm_self_repair_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "churn.csv"
    generate(data_path, n_rows=240)
    spec = CapabilitySpec(raw_description="客户流失预测 self-repair", feature_columns=["age", "region", "login_count_30d", "total_spend"], metric_thresholds={"roc_auc": 0.65})
    plan = AlgorithmPlan("algorithm_logistic_regression", "Logistic Regression", "repair fixture", ["imputation", "one-hot", "scaling"], {"C": 1.0, "max_iter": 500, "scaler": "standard"}, base_algorithm_id="algorithm_logistic_regression")
    algorithm_path = GeneratorAgent(None, "fixture").run(output_dir / "work", spec, plan, allow_llm=False)
    original = algorithm_path.read_text(encoding="utf-8")
    broken = original.replace("def predict(model, test_df", "def predict_broken(model, test_df")
    algorithm_path.write_text(broken, encoding="utf-8")
    (output_dir / "before.py").write_text(broken, encoding="utf-8")
    validator = ValidationRunner(30)
    first = validator.run(algorithm_path, data_path, spec, plan.algorithm_name)
    historical = [{"id": "repair_interface_predict_v1", "task_type": "binary_classification", "domain": "customer_churn", "failure_type": "interface_failure", "triggering_condition": "missing function: predict", "repair_action": "restore exact predict(model, test_df) and preserve train/evaluate", "repair_success": True}]
    retrieved = ExperienceRetriever().retrieve_failures(spec, historical)
    coder_path = "/data/public_checkpoints/huggingface_models/Qwen2.5-Coder-3B-Instruct"
    llm = LocalTransformersLLM(coder_path)
    started = time.perf_counter()
    diagnosis = CriticAgent(llm, "local_coder").run(spec, plan, first, broken, retrieved)
    repair_event = RepairAgent(llm, "local_coder").repair(algorithm_path, json.dumps({"validation": first.to_dict(), "diagnosis": diagnosis}, ensure_ascii=False), 1, retrieved)
    repaired = algorithm_path.read_text(encoding="utf-8")
    (output_dir / "after.py").write_text(repaired, encoding="utf-8")
    second = validator.run(algorithm_path, data_path, spec, plan.algorithm_name, repair_round=1)
    real_llm_success = second.status == "passed" and repair_event.get("provider") == "local_coder" and any("accepted LLM repair" in change for change in repair_event.get("changes", []))
    evidence = {
        "model": Path(coder_path).name, "model_path": coder_path, "real_llm_success": real_llm_success,
        "first_validation": first.to_dict(), "retrieved_experiences": retrieved, "critic_diagnosis": diagnosis,
        "repair_event": repair_event, "second_validation": second.to_dict(), "elapsed_seconds": round(time.perf_counter() - started, 3),
        "artifacts": {"before": "before.py", "after": "after.py", "data": "churn.csv"},
    }
    (output_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"real_llm_success": real_llm_success, "first": first.status, "second": second.status, "model": Path(coder_path).name, "repair_event": repair_event, "elapsed_seconds": evidence["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    return 0 if real_llm_success else 1


if __name__ == "__main__":
    raise SystemExit(main())

