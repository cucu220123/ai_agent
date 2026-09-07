from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from app.config import Settings
from app.workflow import AlgorithmFactoryWorkflow
from scripts.generate_demo_data import generate


def settings(root):
    return Settings(project_root=root, data_dir=root / "data", generated_dir=root / "generated", reports_dir=root / "reports", knowledge_db=root / "k.sqlite", graphml_path=root / "k.graphml", llm_provider="mock", embedding_model_path=None, max_repair_rounds=1, validation_timeout_seconds=30)


def test_measured_task_a_writeback_enters_task_b_planner_and_failure_retrieval(tmp_path):
    factory = AlgorithmFactoryWorkflow(settings(tmp_path))
    received_contexts = []
    original_repair = factory.repair.repair
    def observe_repair(*args, **kwargs):
        received_contexts.append(kwargs["context"])
        return original_repair(*args, **kwargs)
    factory.repair.repair = observe_repair
    assert not factory.store.list_validation_runs()
    data_a = generate(tmp_path / "a.csv", n_rows=240, seed=42)
    a = factory.run("预测客户流失，ROC-AUC 不低于 0.60，输出概率", data_a, inject_repair_failure=True)
    assert a.validation.status == "passed"
    assert len(a.writeback["validation_run_ids"]) >= len(a.candidate_results)
    assert a.writeback["failure_experience_ids"]
    repaired = next(c for c in a.candidate_results if len(c["attempts"]) > 1)
    assert received_contexts[0]["requirement"]["target_column"] == a.spec.target_column
    assert received_contexts[0]["execution_plan"]["algorithm_id"] == repaired["plan"]["algorithm_id"]
    assert received_contexts[0]["requirement"]["dataset_profile"]["rows"] == 240
    assert repaired["attempts"][0]["validation"]["status"] == "failed"
    assert repaired["attempts"][1]["validation"]["status"] == "passed"
    assert repaired["attempts"][1]["parent_version"] == repaired["attempts"][0]["version_id"]
    assert repaired["attempts"][0]["code_hash"] != repaired["attempts"][1]["code_hash"]
    data_b = generate(tmp_path / "b.csv", n_rows=280, seed=43)
    b = factory.run("根据客户行为做流失预测，ROC-AUC 不低于 0.61，优先可解释", data_b)
    retrieved_ids = {case["run_id"] for case in b.knowledge.historical_cases}
    assert a.run_id in retrieved_ids
    assert a.run_id in {case["run_id"] for case in b.knowledge.planning_context["similar_historical_runs"]}
    assert any(a.run_id in plan.evidence_ids for plan in b.plans)
    assert set(a.writeback["failure_experience_ids"]) & {e["id"] for e in b.knowledge.experiences}
    assert all(Path(item["algorithm_path"]).is_file() for c in b.candidate_results for item in c["attempts"])
    assert b.spec.dataset_profile["rows"] == 280


def test_current_nonlinear_data_overrides_historical_linear_champion(tmp_path):
    factory = AlgorithmFactoryWorkflow(settings(tmp_path))
    factory.store.add_validation_run({"run_id": "historical_linear_champion", "algorithm_id": "algorithm_logistic_regression", "status": "passed", "task_type": "binary_classification", "domain": "customer_churn", "target": "churn", "feature_columns": ["age"], "metrics": {"roc_auc": .999}, "runtime_seconds": .1})
    x, y = make_circles(n_samples=360, noise=.08, factor=.35, random_state=19)
    data = tmp_path / "nonlinear.csv"
    pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "churn": y}).to_csv(data, index=False)
    result = factory.run("根据 x1 和 x2 预测客户流失，ROC-AUC 不低于 0.50", data)
    assert result.validation.status == "passed"
    assert result.selected_plan.base_algorithm_id != "algorithm_logistic_regression"
    assert any(c["plan"]["base_algorithm_id"] == "algorithm_logistic_regression" for c in result.candidate_results)
    assert len({c["plan"]["base_algorithm_id"] for c in result.candidate_results}) == 3
