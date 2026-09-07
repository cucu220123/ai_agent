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

