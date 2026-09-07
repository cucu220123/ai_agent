from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec
from app.retrieval.graph import GraphRetriever


def test_relation_path_includes_repair_and_excludes_wrong_task(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite")
    store.upsert_capability({"id": "churn", "name": "客户流失预测", "task_type": "binary_classification", "domain": "customer_churn"})
    store.upsert_algorithm({"id": "algorithm_a", "name": "A", "task_types": ["binary_classification"]})
    store.add_edge("churn", "algorithm_a", "USES_ALGORITHM")
    store.add_validation_run({"run_id": "run_A", "algorithm_id": "algorithm_a", "task_type": "binary_classification", "status": "failed"})
    store.add_experience({"id": "failure_A", "algorithm_id": "algorithm_a", "run_id": "run_A", "task_type": "binary_classification", "failure_type": "interface_failure"})
    store.add_repair_experience({"id": "repair_A", "failure_id": "failure_A", "strategy": "restore predict"})
    store.add_validation_run({"run_id": "unrelated_regression", "algorithm_id": "algorithm_a", "task_type": "regression"})
    found = GraphRetriever(store.graph, store.get_node_payload).retrieve(CapabilitySpec(raw_description="客户流失预测"), hops=3)
    ids = {n["id"] for n in found["nodes"]}
    assert "repair_A" in ids and "unrelated_regression" not in ids
    assert all(len(p["edge_path"]) <= 3 for p in found["paths"])
    candidate = found["serialized"]["candidate_algorithms"][0]
    assert candidate["repair_experiences"][0]["id"] == "repair_A"
    assert any(p["target"] == "run_A" and any(e["relation"] == "VALIDATES" for e in p["edge_path"]) for p in found["paths"])


def test_generic_dependency_hub_does_not_connect_unrelated_algorithm(tmp_path):
    store = KnowledgeStore(tmp_path / "k.sqlite")
    store.upsert_capability({"id": "cap", "name": "churn", "task_type": "binary_classification"})
    for name in ("a", "b"):
        store.upsert_algorithm({"id": name, "name": name})
    store.upsert_knowledge_item("dep", "Dependency", {"name": "sklearn"})
    store.add_edge("cap", "a", "USES_ALGORITHM")
    store.add_edge("a", "dep", "REQUIRES")
    store.add_edge("b", "dep", "REQUIRES")
    result = GraphRetriever(store.graph, store.get_node_payload).retrieve(CapabilitySpec(raw_description="churn"))
    assert "a" in {n["id"] for n in result["nodes"]}
    assert "b" not in {n["id"] for n in result["nodes"]}
