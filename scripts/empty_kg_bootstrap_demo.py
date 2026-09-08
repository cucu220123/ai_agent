"""Extraction/retrieval smoke demo; no algorithm execution or ValidationRunner."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.knowledge.extractor import CapabilityExtractor
from app.knowledge.retriever import RetrieverAgent
from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec
from app.llm.factory import build_llm
from app.config import Settings


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="empty_kg_bootstrap_") as temp:
        root = Path(temp)
        store = KnowledgeStore(root / "knowledge.sqlite", root / "knowledge.graphml")
        before = store.graph_summary()
        # No seed_data/knowledge.json is loaded. Raw business material is the only source.
        provider = os.getenv("BOOTSTRAP_PROVIDER", "mock")
        settings = Settings(llm_provider=provider, local_instruction_model_path=os.getenv("LOCAL_INSTRUCTION_MODEL_PATH", "/data/public_checkpoints/huggingface_models/Qwen2.5-14B-Instruct"), local_model_path=os.getenv("LOCAL_INSTRUCTION_MODEL_PATH", "/data/public_checkpoints/huggingface_models/Qwen2.5-14B-Instruct"), embedding_model_path=None)
        llm = build_llm(settings) if provider != "mock" else None
        extractor = CapabilityExtractor(llm, provider)
        extractor.ingest(PROJECT_ROOT / "data/business_material.md", store)
        extractor.ingest(PROJECT_ROOT / "examples/self_repair_demo/algorithm.py", store)
        extractor.ingest(PROJECT_ROOT / "reports/116eedc944ea.json", store) if (PROJECT_ROOT / "reports/116eedc944ea.json").exists() else None
        after = store.graph_summary()
        spec = CapabilitySpec(raw_description="客户流失预测", domain="customer_churn", capability_name="客户流失预测", task_type="binary_classification", target_column="churn", feature_columns=["age", "login_count_30d"], metrics=["roc_auc"])
        retrieval = RetrieverAgent(store).run(spec)
        evidence = {"nodes_before": before["nodes"], "edges_before": before["edges"], "nodes_after": after["nodes"], "edges_after": after["edges"], "extracted_sources": store.list_knowledge_items(20), "retrieval_trace": retrieval.retrieval_trace, "retrieved_subgraph": retrieval.graph_evidence, "candidate_algorithms": retrieval.algorithms, "selected_algorithm": retrieval.algorithms[0] if retrieval.algorithms else None, "algorithm_retrieval_succeeded": bool(retrieval.algorithms)}
        target = PROJECT_ROOT / "docs/evidence/empty_kg_bootstrap.json"
        target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"nodes_before": evidence["nodes_before"], "edges_before": evidence["edges_before"], "nodes_after": evidence["nodes_after"], "edges_after": evidence["edges_after"], "retrieved_nodes": len(retrieval.graph_evidence.get("nodes", []))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
