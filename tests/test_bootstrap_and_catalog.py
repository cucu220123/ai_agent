from app.knowledge.extractor import CapabilityExtractor
from app.knowledge.store import KnowledgeStore


def test_empty_graph_bootstrap_has_no_seed_requirement(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite", tmp_path / "k.graphml")
    before = store.graph_summary()
    assert before["nodes"] == 0
    CapabilityExtractor().ingest(__import__("pathlib").Path(__file__).parents[1] / "data/business_material.md", store)
    after = store.graph_summary()
    assert after["nodes"] > 0
    assert not any("churn" in node for node in store.graph.nodes if node.startswith("cap_churn_prediction_v1"))


def test_generic_catalog_has_no_domain_dataset_or_capability(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite", tmp_path / "k.graphml")
    store.ensure_catalog_nodes()
    assert "cap_churn_prediction_v1" not in store.graph.nodes
    assert "dataset_churn_demo" not in store.graph.nodes

