from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents.critic_agent import CriticAgent
from app.agents.generator_agent import GeneratorAgent
from app.agents.repair_agent import RepairAgent
from app.models import AlgorithmPlan, CapabilitySpec
from app.validation.runner import ValidationRunner
from scripts.generate_demo_data import generate


def run_self_repair_demo(root: str | Path, llm=None, provider_name: str = "mock") -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "churn.csv"
    generate(data_path, n_rows=240)
    spec = CapabilitySpec(raw_description="self repair demo", feature_columns=["age"], metric_thresholds={"roc_auc": 0.75})
    plan = AlgorithmPlan("algorithm_logistic_regression", "Logistic Regression", "demo", ["imputation"], {"C": 1.0, "max_iter": 500})
    generator = GeneratorAgent(llm, provider_name)
    path = generator.run(root, spec, plan)
    before = path.read_text(encoding="utf-8")
    # Deliberately create an interface failure so the Critic/Repair/Validator loop is observable.
    broken = before.replace("def predict(model, test_df", "def predict_broken(model, test_df")
    path.write_text(broken, encoding="utf-8")
    static_result = ValidationRunner(30).run(path, data_path, spec, plan.algorithm_name, repair_round=0)
    critic = CriticAgent(llm, provider_name)
    diagnosis = critic.run(spec, plan, static_result, broken)
    repaired_event = RepairAgent(llm, provider_name).repair(path, json.dumps(diagnosis, ensure_ascii=False) + "\n" + "\n".join(static_result.errors), 1)
    after = path.read_text(encoding="utf-8")
    second_validation = ValidationRunner(30).run(path, data_path, spec, plan.algorithm_name, repair_round=1)
    return {"before_code": before, "broken_code": broken, "first_validation": static_result.to_dict(), "diagnosis": diagnosis, "repair_event": repaired_event, "after_code": after, "second_validation": second_validation.to_dict(), "repair_status": "passed" if second_validation.status == "passed" else "failed"}
