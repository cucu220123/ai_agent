import pytest
from pydantic import ValidationError

from app.agents.knowledge_extraction_agent import KnowledgeExtractionContract


def test_relation_enum_rejects_magic_relation():
    with pytest.raises(ValidationError):
        KnowledgeExtractionContract.model_validate({"entities": [{"id": "a", "type": "Algorithm", "name": "A", "evidence_span": "A", "confidence": 0.9}, {"id": "b", "type": "Task", "name": "B", "evidence_span": "B", "confidence": 0.9}], "relations": [{"source": "a", "relation": "MAGIC_RELATION", "target": "b", "evidence_span": "A B", "confidence": 0.9}], "summary": "x"})
