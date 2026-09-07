import pytest
from app.agents.knowledge_extraction_agent import KnowledgeExtractionAgent, KnowledgeExtractionContract


def contract(a="a", b="b"):
    return KnowledgeExtractionContract.model_validate({"entities": [
        {"id": a, "type": "Capability", "name": "forecast", "evidence_span": "forecast uses linear", "confidence": 0.9},
        {"id": b, "type": "Algorithm", "name": "linear", "evidence_span": "linear", "confidence": 0.9}],
        "relations": [{"source": a, "target": b, "relation": "USES_ALGORITHM", "evidence_span": "forecast uses linear", "confidence": 0.9}], "summary": "forecast"})


def test_duplicate_chunk_ids_rewritten_without_dangling_edges():
    merged = KnowledgeExtractionAgent._merge([contract(), contract("x", "y")], "source.md")
    ids = {e["id"] for e in merged["entities"]}
    assert len(ids) == 2 and len(merged["relations"]) == 1
    assert all(r["source"] in ids and r["target"] in ids for r in merged["relations"])
    assert len(merged["entities"][0]["provenance_records"]) == 2


def test_fabricated_source_span_is_rejected():
    errors = KnowledgeExtractionAgent._quality_gate(contract(), {}, {"text": "unrelated text"})
    assert any("ungrounded" in e for e in errors)


def test_metrics_are_chunk_local():
    errors = KnowledgeExtractionAgent._quality_gate(contract(), {"metrics": ["ROC-AUC"]}, {"text": "forecast uses linear"})
    assert not errors


def test_weighted_f1_qualifier_is_not_discarded_or_falsely_rejected():
    extracted = KnowledgeExtractionContract.model_validate({"entities": [{"id": "m", "type": "Metric", "name": "weighted F1", "evidence_span": "weighted F1", "confidence": 1}], "relations": [], "summary": "weighted F1"})
    assert not KnowledgeExtractionAgent._quality_gate(extracted, {"metrics": ["f1"]}, {"text": "Report weighted F1"})
    assert extracted.entities[0].name == "weighted F1"
