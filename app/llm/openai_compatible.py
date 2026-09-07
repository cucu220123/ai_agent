from __future__ import annotations

import time
import re
from urllib.parse import urlsplit
from typing import Any

from app.llm.generation import resolve_generation_config
from app.llm.security import load_credentials, register_secret, sanitize

load_openai_settings = load_credentials


class OpenAICompatibleLLM:
    """JSON mode/schema negotiation with bounded transport retries and safe errors."""

    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o-mini", timeout: float = 180.0):
        from openai import OpenAI
        register_secret(api_key)
        self.model = model
        self.last_provider = "local_openai_compatible" if urlsplit(base_url).hostname in {"localhost", "127.0.0.1", "::1"} else "openai_compatible"
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.last_usage: dict[str, int] = {}
        self.last_retry_count = 0
        self.last_generation: dict[str, Any] = {}
        self.schema_supported: bool | None = None

    def complete(self, system: str, user: str, purpose: str = "general", generation_config: dict[str, Any] | None = None) -> str:
        config = resolve_generation_config(purpose, generation_config)
        kwargs: dict[str, Any] = {
            "model": self.model, "temperature": config.temperature,
            "max_tokens": config.max_new_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            if config.json_schema and self.schema_supported is not False:
                kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": purpose, "schema": config.json_schema, "strict": False}}
        self.last_usage = {}
        self.last_retry_count = 0
        self.last_generation = {}
        budget_adjustments = []
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                message = str(exc).lower()
                if status in {400, 422} and kwargs.get("response_format", {}).get("type") == "json_schema" and any(word in message for word in ("schema", "response_format", "unsupported")):
                    self.schema_supported = False
                    kwargs["response_format"] = {"type": "json_object"}
                elif status == 400 and (available := self._context_output_budget(message, kwargs["max_tokens"])) is not None and attempt < 2:
                    budget_adjustments.append({"requested": kwargs["max_tokens"], "effective": available, "reason": "provider_reported_context_limit"})
                    kwargs["max_tokens"] = available
                elif status in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(min(2.0, 0.5 * 2 ** attempt))
                else:
                    raise RuntimeError(sanitize(f"{type(exc).__name__}: {exc}")) from None
                self.last_retry_count += 1
        else:
            raise RuntimeError("provider retries exhausted")
        usage = getattr(response, "usage", None)
        if usage:
            self.last_usage = {k: int(getattr(usage, k)) for k in ("prompt_tokens", "completion_tokens", "total_tokens") if getattr(usage, k, None) is not None}
        choice = response.choices[0]
        reason = getattr(choice, "finish_reason", None)
        self.last_generation = {"purpose": purpose, "max_new_tokens": kwargs["max_tokens"], "requested_max_new_tokens": config.max_new_tokens, "budget_adjustments": budget_adjustments, "finish_reason": reason, "truncated": reason == "length", "eos_reached": reason in {"stop", "eos"}, "structured_transport": kwargs.get("response_format", {}).get("type", "text"), "application_schema_validation": bool(config.json_schema)}
        if reason == "length":
            raise RuntimeError("LLM output truncated at configured token limit")
        return choice.message.content or ""



    @staticmethod
    def _context_output_budget(message: str, requested: int) -> int | None:
        """Use exact provider counts; never silently drop requirement/evidence text.

        vLLM reports both counts on an oversized output reservation. Unknown
        error formats remain errors. A truncated response is still rejected.
        """
        supplied = re.search(r"passed (\d+) input tokens", message)
        capacity = re.search(r"context length is only (\d+) tokens", message)
        if not supplied or not capacity:
            return None
        available = int(capacity.group(1)) - int(supplied.group(1)) - 128
        return available if 256 <= available < requested else None
