from app.config import Settings
from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec
from app.experience.closed_loop import ClosedLoopAnalyzer


def test_written_run_is_retrievable(tmp_path):
    settings = Settings(project_root=tmp_path, knowledge_db=tmp_path / "k.sqlite", graphml_path=tmp_path / "k.graphml", data_dir=tmp_path / "data", generated_dir=tmp_path / "gen", reports_dir=tmp_path / "reports")
    store = KnowledgeStore(settings.knowledge_db, settings.graphml_path)
    store.seed_from_json(__import__("pathlib").Path(__file__).parents[1] / "app/knowledge/seed_data/knowledge.json")
    store.ensure_catalog_nodes()
    spec = CapabilitySpec(raw_description="churn", domain="customer_churn", target_column="churn", feature_columns=["age"])
    before = ClosedLoopAnalyzer(store).before_after(spec)
    store.add_validation_run({"run_id": "run_closed", "capability_id": "cap_churn_prediction_v1", "algorithm_id": "algorithm_logistic_regression", "status": "passed", "task_type": "binary_classification", "domain": "customer_churn", "target": "churn", "feature_columns": ["age"], "metrics": {"roc_auc": 0.91}})
    after = ClosedLoopAnalyzer(store).after(spec, "run_closed")
    assert not any(x.get("run_id") == "run_closed" for x in before["before"]["historical_cases"])
    assert "run_closed" in after["after"]["matched_run_ids"]

