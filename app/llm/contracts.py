from __future__ import annotations

import json
import time
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
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0

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


def extract_python_code(text: str) -> str | None:
    """Extract a Python program from a fenced or plain model response."""
    text = (text or "").strip()
    if not text:
        return None
    if "```" in text:
        parts = text.split("```")
        for idx in range(1, len(parts), 2):
            block = parts[idx].strip()
            if block.startswith("python"):
                block = block[6:].lstrip("\n ")
            if "def train" in block and "def predict" in block:
                return block
    if "def train" in text and "def predict" in text and "def evaluate" in text:
        return text[text.find("from ") if "from " in text else 0 :]
    return None


def complete_with_trace(provider: Any, provider_name: str, model: str | None, purpose: str, system: str, user: str, generation_config: dict[str, Any] | None = None) -> tuple[str, LLMTrace]:
    trace = LLMTrace(provider=provider_name, model=model, purpose=purpose, status="started")
    started = time.perf_counter()
    try:
        response = provider.complete(system, user, purpose=purpose, generation_config=generation_config) or ""
        trace.status = "ok"
        trace.provider = getattr(provider, "last_provider", provider_name)
        trace.model = getattr(provider, "model", model)
        trace.response_preview = response[:1000]
        trace.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        trace.token_usage = getattr(provider, "last_usage", {}) or {}
        trace.retry_count = int(getattr(provider, "last_retry_count", 0) or 0)
        trace.generation = getattr(provider, "last_generation", {}) or {}
        return response, trace
    except Exception as exc:
        trace.status = "fallback"
        trace.error = f"{type(exc).__name__}: {exc}"[:1000]
        trace.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return "", trace
