from __future__ import annotations

from app.experience.retriever import ExperienceRetriever
from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec, KnowledgeContext
from app.retrieval.graph import GraphRetriever
from app.retrieval.semantic import HybridSemanticRetriever
from app.retrieval.fusion import rerank_evidence


class RetrieverAgent:
    """Hybrid GraphRAG retriever: entity-linking graph traversal + semantic evidence + cases."""

    def __init__(self, store: KnowledgeStore, embedding_model_path: str | None = None):
        self.store = store
        self.graph = GraphRetriever(store.graph, store.get_node_payload)
        self.semantic = HybridSemanticRetriever(embedding_model_path)
        self.experience = ExperienceRetriever()

    def run(self, spec: CapabilitySpec) -> KnowledgeContext:
        query = " ".join([spec.capability_name, spec.domain, spec.task_type, spec.data_type, spec.target_column, *spec.feature_columns, *spec.metrics, *spec.constraints])
        graph_evidence = self.graph.retrieve(spec, hops=3, limit=64)
        documents = self.store.list_knowledge_items(200) + self.store.recent_experiences(100) + self.store.list_validation_runs(100)
        semantic_evidence = self.semantic.retrieve(spec, documents, limit=8)
        semantic_evidence = rerank_evidence(spec, semantic_evidence, graph_evidence)
        historical_cases = self.experience.retrieve(spec, self.store.list_validation_runs(100), limit=12)
        algorithms = self.store.list_algorithms()
        relevant_experiences = self.experience.retrieve_failures(spec, self.store.recent_experiences(100), limit=8)
        graph_algorithms = graph_evidence.get("serialized", {}).get("candidate_algorithms", [])
        if graph_algorithms:
            selected_ids = {a.get("algorithm_id") for a in graph_algorithms}
            algorithms = [a for a in algorithms if a.get("id") in selected_ids] or algorithms
        return KnowledgeContext(
            retrieval_query=query,
            capabilities=self.store.list_capabilities(),
            algorithms=algorithms,
            experiences=relevant_experiences,
            graph_evidence=graph_evidence,
            semantic_evidence=semantic_evidence,
            historical_cases=historical_cases,
            retrieval_trace={
                "strategy": "hybrid_graph_rag",
                "graph_anchors": len(graph_evidence.get("anchors", [])),
                "graph_nodes": len(graph_evidence.get("nodes", [])),
                "graph_edges": len(graph_evidence.get("edges", [])),
                "semantic_documents": len(semantic_evidence),
                "semantic_backend": self.semantic.last_backend,
                "semantic_fallback_error": self.semantic.last_error,
                "historical_cases": len(historical_cases),
                "fusion": "semantic+graph_distance+task+recency+validation_quality",
            },
        )
