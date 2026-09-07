from __future__ import annotations

import time
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
        self.last_provider = "openai_compatible"
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
        self.last_generation = {"purpose": purpose, "max_new_tokens": config.max_new_tokens, "finish_reason": reason, "truncated": reason == "length", "eos_reached": reason in {"stop", "eos"}, "structured_transport": kwargs.get("response_format", {}).get("type", "text"), "application_schema_validation": bool(config.json_schema)}
        if reason == "length":
            raise RuntimeError("LLM output truncated at configured token limit")
        return choice.message.content or ""


