"""A single auditable invocation boundary shared by every LLM-backed agent."""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any

from app.llm.secrets import sanitize


class LLMInvocationError(RuntimeError):
    pass


class TracedLLM:
    def __init__(self, provider: Any, provider_name: str):
        self.provider = provider
        self.provider_name = provider_name
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def complete(self, system: str, user: str, purpose: str = "general", generation_config: dict[str, Any] | None = None) -> str:
        started = time.perf_counter()
        record = {"call_id": f"llm_{len(self.calls) + 1:04d}", "purpose": purpose, "timestamp": datetime.now(timezone.utc).isoformat(), "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest(), "prompt_chars": len(system) + len(user)}
        try:
            result = self.provider.complete(system, user, purpose=purpose, generation_config=generation_config)
            record.update(status="ok", response_sha256=hashlib.sha256(result.encode()).hexdigest())
            return result
        except Exception as exc:
            record.update(status="error", error=sanitize(f"{type(exc).__name__}: {exc}"))
            raise LLMInvocationError(record["error"]) from None
        finally:
            record.update(provider=getattr(self.provider, "last_provider", self.provider_name), model=getattr(self.provider, "model", None), latency_ms=round((time.perf_counter() - started) * 1000, 2), token_usage=getattr(self.provider, "last_usage", {}), retry_count=getattr(self.provider, "last_retry_count", 0), generation=getattr(self.provider, "last_generation", {}))
            self.calls.append(sanitize(record))
