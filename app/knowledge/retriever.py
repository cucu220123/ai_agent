from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec, KnowledgeContext


class RetrieverAgent:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def run(self, spec: CapabilitySpec) -> KnowledgeContext:
        query = " ".join([spec.capability_name, spec.task_type, spec.target_column, *spec.feature_columns, *spec.metrics])
        result = self.store.search(query)
        return KnowledgeContext(retrieval_query=query, **result)

