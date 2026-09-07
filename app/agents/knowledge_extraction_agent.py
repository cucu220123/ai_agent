from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.llm.contracts import extract_json_object


class KnowledgeExtractionAgent:
    """Convert source artifacts into provenance-preserving structured knowledge."""

    def __init__(self, llm=None, provider_name: str = "mock"):
        self.llm = llm
        self.provider_name = provider_name

    def run(self, source_path: str | Path, deterministic_facts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        source_path = str(source_path)
        trace = {"provider": self.provider_name, "status": "deterministic_fallback", "source": source_path}
        if self.llm is not None and self.provider_name != "mock":
            try:
                raw = self.llm.complete(
                    "你是 KnowledgeExtractionAgent。只输出 JSON。不要杜撰材料中不存在的指标。",
                    json.dumps({"source": source_path, "deterministic_facts": deterministic_facts, "required_schema": {"capabilities": [], "tasks": [], "algorithms": [], "datasets": [], "features": [], "preprocessing": [], "metrics": [], "constraints": [], "dependencies": [], "hyperparameters": [], "failure_cases": [], "optimization_experiences": [], "validation_results": [], "provenance": {"source": source_path, "evidence_spans": []}}}, ensure_ascii=False),
                ) or ""
                parsed = extract_json_object(raw)
                if parsed:
                    parsed.setdefault("provenance", {"source": source_path, "evidence_spans": []})
                    parsed["provenance"].setdefault("source", source_path)
                    trace.update({"status": "ok", "response_preview": raw[:1000]})
                    return parsed, trace
                trace["error"] = "LLM response was not a JSON object"
            except Exception as exc:
                trace["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        facts = dict(deterministic_facts)
        facts["provenance"] = {"source": source_path, "evidence_spans": []}
        return facts, trace

