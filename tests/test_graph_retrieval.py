from app.config import Settings
from app.knowledge.store import KnowledgeStore
from app.knowledge.retriever import RetrieverAgent
from app.models import CapabilitySpec


def test_graph_retrieval_returns_subgraph(tmp_path):
    settings = Settings(project_root=tmp_path, knowledge_db=tmp_path / "k.sqlite", graphml_path=tmp_path / "k.graphml", data_dir=tmp_path / "data", generated_dir=tmp_path / "gen", reports_dir=tmp_path / "reports")
    store = KnowledgeStore(settings.knowledge_db, settings.graphml_path)
    store.seed_from_json(__import__("pathlib").Path(__file__).parents[1] / "app/knowledge/seed_data/knowledge.json")
    store.ensure_catalog_nodes()
    context = RetrieverAgent(store).run(CapabilitySpec(raw_description="churn", domain="customer_churn", target_column="churn", feature_columns=["age"], metrics=["roc_auc"]))
    assert context.retrieval_trace["strategy"] == "hybrid_graph_rag"
    assert context.graph_evidence["nodes"]
    assert context.graph_evidence["edges"]
    assert context.graph_evidence["serialized"]["task"]["task_type"] == "binary_classification"


def test_graph_serializes_full_validation_metrics(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite", tmp_path / "k.graphml")
    store.seed_from_json(__import__("pathlib").Path(__file__).parents[1] / "app/knowledge/seed_data/knowledge.json")
    store.ensure_catalog_nodes()
    store.add_validation_run({"run_id": "run_metric_091", "capability_id": "cap_churn_prediction_v1", "algorithm_id": "algorithm_logistic_regression", "status": "passed", "task_type": "binary_classification", "domain": "customer_churn", "target": "churn", "feature_columns": ["age"], "metrics": {"roc_auc": 0.91, "f1": 0.77}, "runtime_seconds": 1.2, "timestamp": "2026-09-07T00:00:00Z"})
    context = RetrieverAgent(store).run(CapabilitySpec(raw_description="churn", domain="customer_churn", target_column="churn", feature_columns=["age"], metrics=["roc_auc"]))
    candidates = context.graph_evidence["serialized"]["candidate_algorithms"]
    runs = [run for candidate in candidates for run in candidate["historical_runs"] if run.get("run_id") == "run_metric_091"]
    assert runs
    assert runs[0]["metrics"]["roc_auc"] == 0.91


def test_graph_retrieval_is_not_churn_hardcoded(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite", tmp_path / "k.graphml")
    store.upsert_capability({"id": "cap_inventory", "name": "库存预测", "domain": "inventory", "task_type": "regression", "target": "demand"})
    store.upsert_algorithm({"id": "algorithm_inventory_rf", "name": "Inventory RF", "task_types": ["regression"], "historical_metrics": {"r2": 0.8}})
    store.add_edge("cap_inventory", "algorithm_inventory_rf", "USES_ALGORITHM")
    context = RetrieverAgent(store).run(CapabilitySpec(raw_description="预测库存需求", domain="inventory", capability_name="库存预测", task_type="regression", target_column="demand", metrics=["r2"], candidate_algorithms=[]))
    ids = {node["id"] for node in context.graph_evidence["nodes"]}
    assert "cap_inventory" in ids
    assert "algorithm_inventory_rf" in ids
