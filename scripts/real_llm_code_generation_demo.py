from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.generator_agent import GeneratorAgent
from app.agents.critic_agent import CriticAgent
from app.agents.repair_agent import RepairAgent
from app.experience.retriever import ExperienceRetriever
from app.llm.local_transformers import LocalTransformersLLM
from app.models import AlgorithmPlan, CapabilitySpec
from app.validation.runner import ValidationRunner


def main() -> int:
    output_dir = PROJECT_ROOT / "examples/real_llm_code_generation_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    coder_path = "/data/public_checkpoints/huggingface_models/Qwen2.5-Coder-3B-Instruct"
    llm = LocalTransformersLLM(coder_path)
    spec = CapabilitySpec(raw_description="客户流失预测", domain="customer_churn", capability_name="客户流失预测", feature_columns=["age", "region", "login_count_30d", "total_spend", "complaint_count", "membership_level", "tenure_months"], input_schema={"age": "int", "region": "category", "login_count_30d": "float", "total_spend": "float", "complaint_count": "int", "membership_level": "category", "tenure_months": "int"}, metric_thresholds={"roc_auc": 0.75})
    plan = AlgorithmPlan("algorithm_logistic_regression__real_llm", "Logistic Regression", "benchmark-selected coder demo", ["imputation", "one-hot", "standard scaling"], {"C": 1.0, "max_iter": 500, "class_weight": "balanced", "threshold": 0.4}, base_algorithm_id="algorithm_logistic_regression")
    started = time.perf_counter()
    algorithm_path = GeneratorAgent(llm, "local_coder").run(output_dir, spec, plan, allow_llm=True)
    metadata = json.loads((output_dir / "algorithm_meta.json").read_text(encoding="utf-8"))
    initial_code = algorithm_path.read_text(encoding="utf-8")
    (output_dir / "initial_llm_code.py").write_text(initial_code, encoding="utf-8")
    validation = ValidationRunner(60).run(algorithm_path, PROJECT_ROOT / "data/churn_demo.csv", spec, plan.algorithm_name)
    initial_validation = validation.to_dict()
    repair_events = []
    experiences = [
        {"id": "repair_invalid_estimator_parameter", "task_type": "binary_classification", "domain": "customer_churn", "failure_type": "runtime_failure", "triggering_condition": "unexpected keyword argument in sklearn estimator", "repair_action": "remove unsupported estimator kwargs such as threshold; threshold belongs in prediction logic", "repair_success": True},
        {"id": "repair_column_transformer_target_leak", "task_type": "binary_classification", "domain": "customer_churn", "failure_type": "runtime_failure", "triggering_condition": "A given column is not a column of the dataframe", "repair_action": "first set X=train_df.drop(target); derive numeric_cols and categorical_cols from X, never train_df; categorical branch must be Pipeline(SimpleImputer, OneHotEncoder); predict receives features only and must not drop target/churn", "repair_success": True},
    ]
    retrieved = ExperienceRetriever().retrieve_failures(spec, experiences)
    for round_no in range(1, 5):
        if validation.status == "passed":
            break
        diagnosis = CriticAgent(llm, "local_coder").run(spec, plan, validation, algorithm_path.read_text(encoding="utf-8"), retrieved)
        event = RepairAgent(llm, "local_coder").repair(algorithm_path, json.dumps({"validation": validation.to_dict(), "diagnosis": diagnosis}, ensure_ascii=False), round_no, retrieved)
        event["diagnosis"] = diagnosis
        repair_events.append(event)
        validation = ValidationRunner(60).run(algorithm_path, PROJECT_ROOT / "data/churn_demo.csv", spec, plan.algorithm_name, round_no)
    (output_dir / "final_llm_code.py").write_text(algorithm_path.read_text(encoding="utf-8"), encoding="utf-8")
    llm_generation_accepted = metadata["generation"].get("status") == "llm_code_accepted"
    repaired_by_llm = any(event.get("provider") == "local_coder" and any("accepted LLM repair" in change for change in event.get("changes", [])) for event in repair_events)
    llm_success = llm_generation_accepted and validation.status == "passed" and (initial_validation["status"] == "passed" or repaired_by_llm)
    evidence = {"model": Path(coder_path).name, "llm_success": llm_success, "raw_llm_executable": initial_validation["status"] == "passed", "final_llm_pipeline_executable": validation.status == "passed", "generation": metadata["generation"], "initial_validation": initial_validation, "retrieved_experiences": retrieved, "repair_events": repair_events, "validation": validation.to_dict(), "elapsed_seconds": round(time.perf_counter() - started, 3), "code_path": "final_llm_code.py"}
    (output_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"llm_success": llm_success, "model": evidence["model"], "generation": evidence["generation"], "validation_status": validation.status, "errors": validation.errors, "metrics": validation.metrics, "elapsed_seconds": evidence["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    return 0 if llm_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
