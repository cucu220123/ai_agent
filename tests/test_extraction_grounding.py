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


def test_report_endpoint_error_is_actionable_and_retry_keeps_provenance(tmp_path):
    import json
    from app.knowledge.extractor import CapabilityExtractor
    report = tmp_path / "experiment.json"
    report.write_text(json.dumps({"run_id": "Run_X", "algorithm": "Logistic Regression", "validation": {"metrics": {"roc_auc": 0.87}, "status": "passed"}}))
    facts = CapabilityExtractor().extract_report(report)
    assert facts["algorithm"] == "Logistic Regression"
    entities = [
        {"id": "run1", "type": "ValidationRun", "name": "Run_X", "properties": {"metrics": {"roc_auc": 0.87}}, "evidence_span": "Run_X", "confidence": 1},
        {"id": "algorithm1", "type": "Algorithm", "name": "Logistic Regression", "evidence_span": "Logistic Regression", "confidence": 1},
        {"id": "m1", "type": "Metric", "name": "roc_auc", "evidence_span": "roc_auc", "confidence": 1},
    ]
    class RetryModel:
        def __init__(self): self.payloads = []
        def complete(self, system, user, **kwargs):
            self.payloads.append(json.loads(user))
            return json.dumps({"entities": entities, "relations": [{"source": "missing_run" if len(self.payloads) == 1 else "run1", "target": "algorithm1", "relation": "VALIDATES", "evidence_span": "Logistic Regression", "confidence": 1}], "summary": "Observed experiment"})
    model = RetryModel()
    result, trace = KnowledgeExtractionAgent(model, "openai").run(report, facts)
    assert trace["status"] == "ok" and len(model.payloads) == 2
    assert "missing_run" in model.payloads[1]["validation_error"]
    assert "run1" in model.payloads[1]["validation_error"]
    assert result["entities"][0]["properties"]["metrics"]["roc_auc"] == 0.87
    assert result["relations"][0]["provenance"]["evidence_span"] == "Logistic Regression"
    assert trace["chunks"][0]["attempts"][0]["raw_response"]


def test_same_experiment_label_in_distinct_sources_does_not_overwrite(tmp_path):
    from app.knowledge.extractor import CapabilityExtractor
    from app.knowledge.store import KnowledgeStore
    store = KnowledgeStore(tmp_path / 'kg.sqlite', tmp_path / 'kg.graphml')
    extractor = CapabilityExtractor()
    for source, score in [('source_a', 0.81), ('source_b', 0.91)]:
        extractor._materialize({'source_sha256': source, 'extraction_trace': {'status': 'ok'}, 'provenance': {'source': source}, 'entities': [{'id': 'run1', 'type': 'ValidationRun', 'name': 'Validation Run', 'properties': {'metrics': {'roc_auc': score}, 'status': 'passed'}, 'evidence_span': 'Validation Run', 'confidence': 1}], 'relations': []}, source, store)
    runs = store.list_validation_runs(10)
    assert len(runs) == 2
    assert {r['metrics']['roc_auc'] for r in runs} == {0.81, 0.91}
    assert len({r['run_id'] for r in runs}) == 2
