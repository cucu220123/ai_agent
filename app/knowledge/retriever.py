from __future__ import annotations

from app.experience.retriever import ExperienceRetriever
from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec, KnowledgeContext
from app.retrieval.graph import GraphRetriever
from app.retrieval.semantic import SemanticRetriever


class RetrieverAgent:
    """Hybrid GraphRAG retriever: entity-linking graph traversal + semantic evidence + cases."""

    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.graph = GraphRetriever(store.graph)
        self.semantic = SemanticRetriever()
        self.experience = ExperienceRetriever()

    def run(self, spec: CapabilitySpec) -> KnowledgeContext:
        query = " ".join([spec.capability_name, spec.domain, spec.task_type, spec.data_type, spec.target_column, *spec.feature_columns, *spec.metrics, *spec.constraints])
        graph_evidence = self.graph.retrieve(spec, hops=2, limit=48)
        documents = self.store.list_knowledge_items(200) + self.store.recent_experiences(100) + self.store.list_validation_runs(100)
        semantic_evidence = self.semantic.retrieve(spec, documents, limit=8)
        historical_cases = self.experience.retrieve(spec, self.store.list_validation_runs(100), limit=12)
        algorithms = self.store.list_algorithms()
        graph_algorithms = graph_evidence.get("serialized", {}).get("candidate_algorithms", [])
        if graph_algorithms:
            selected_ids = {a.get("algorithm_id") for a in graph_algorithms}
            algorithms = [a for a in algorithms if a.get("id") in selected_ids] or algorithms
        return KnowledgeContext(
            retrieval_query=query,
            capabilities=self.store.list_capabilities(),
            algorithms=algorithms,
            experiences=self.store.recent_experiences(20),
            graph_evidence=graph_evidence,
            semantic_evidence=semantic_evidence,
            historical_cases=historical_cases,
            retrieval_trace={
                "strategy": "hybrid_graph_rag",
                "graph_anchors": len(graph_evidence.get("anchors", [])),
                "graph_nodes": len(graph_evidence.get("nodes", [])),
                "graph_edges": len(graph_evidence.get("edges", [])),
                "semantic_documents": len(semantic_evidence),
                "historical_cases": len(historical_cases),
            },
        )

