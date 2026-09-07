from app.knowledge.extractor import CapabilityExtractor
from app.knowledge.store import KnowledgeStore


def test_feature_target_metric_roles_are_distinct(tmp_path):
    source = tmp_path / "material.md"
    source.write_text("业务目标：预测 churn 是否流失。\n| age | 数值 | 年龄 |\n| churn | 0/1 | 流失标签 |\nROC-AUC", encoding="utf-8")
    store = KnowledgeStore(tmp_path / "k.sqlite", tmp_path / "k.graphml")
    result = CapabilityExtractor().ingest(source, store)
    # Deterministic fallback identifies metrics/features; target is injected as a separate semantic role.
    node_types = {node_id: attrs.get("type") for node_id, attrs in store.graph.nodes(data=True)}
    assert any(node_type == "Feature" and "age" in node_id for node_id, node_type in node_types.items())
    assert any(node_type == "Target" and "churn" in node_id for node_id, node_type in node_types.items())
    assert any(node_type == "Metric" and "roc_auc" in node_id for node_id, node_type in node_types.items())
    assert node_types.get("feature_churn") != "Feature"

