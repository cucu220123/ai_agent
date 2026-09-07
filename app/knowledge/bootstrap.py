"""Content-addressed knowledge ingestion in the normal workflow, not only a CLI demo."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.knowledge.extractor import CapabilityExtractor
from app.knowledge.store import KnowledgeStore


class KnowledgeBootstrapper:
    def __init__(self, store: KnowledgeStore, llm: Any, provider: str):
        self.store = store
        self.extractor = CapabilityExtractor(llm, provider)
        self.provider = provider

    def run(self, sources: list[Path]) -> list[dict[str, Any]]:
        previous = self.store.list_knowledge_items(1000)
        results = []
        for source in sources:
            if not source.is_file():
                continue
            if source.name.lower().startswith(("secret", ".env")):
                raise ValueError("credential files cannot be ingested")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            match = next((item for item in previous if item.get("source_sha256") == digest and (self.provider == "mock" or item.get("extraction_trace", {}).get("status") == "ok")), None)
            if match:
                results.append({"source": str(source), "source_id": match["id"], "status": "cached", "extraction_status": match.get("extraction_trace", {}).get("status"), "source_sha256": digest})
            else:
                result = self.extractor.ingest(source, self.store)
                results.append({"source": str(source), "source_id": result["id"], "status": "extracted", "extraction_status": result["extraction_trace"]["status"], "source_sha256": digest, "trace": result["extraction_trace"]})
        return results
