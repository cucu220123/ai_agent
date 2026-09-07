from app.knowledge.extractor import CapabilityExtractor
from app.knowledge.store import KnowledgeStore


def test_extracted_structured_items_have_provenance(tmp_path):
    source = tmp_path / "material.md"
    source.write_text("# Customer churn\nROC-AUC F1\n| age | numeric |", encoding="utf-8")
    store = KnowledgeStore(tmp_path / "k.sqlite", tmp_path / "k.graphml")
    result = CapabilityExtractor().ingest(source, store)
    assert result["provenance"]["source"] == str(source)
    assert store.list_knowledge_items(10)
    assert store.graph_summary()["node_types"]["SourceDocument"] == 1

