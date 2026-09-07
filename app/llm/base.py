from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    def complete(self, system: str, user: str, purpose: str = "general", generation_config: dict[str, Any] | None = None) -> str: ...
