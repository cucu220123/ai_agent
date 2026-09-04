from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMTrace:
    provider: str
    model: str | None = None
    status: str = "not_called"
    purpose: str = ""
    response_preview: str = ""
    error: str | None = None
    structured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object even when a model wraps it in Markdown fences."""
    text = (text or "").strip()
    candidates = [text]
    if "```" in text:
        candidates.extend(part.strip() for part in text.split("```") if part.strip())
    for candidate in candidates:
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    value = json.loads(candidate[start : end + 1])
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    continue
    return None


def complete_with_trace(provider: Any, provider_name: str, model: str | None, purpose: str, system: str, user: str) -> tuple[str, LLMTrace]:
    trace = LLMTrace(provider=provider_name, model=model, purpose=purpose, status="started")
    try:
        response = provider.complete(system, user) or ""
        trace.status = "ok"
        trace.response_preview = response[:1000]
        return response, trace
    except Exception as exc:
        trace.status = "fallback"
        trace.error = f"{type(exc).__name__}: {exc}"[:1000]
        return "", trace

