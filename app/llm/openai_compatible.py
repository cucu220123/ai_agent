from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from app.llm.generation import resolve_generation_config


def load_openai_settings(secret_file: str | Path | None = None) -> dict[str, str]:
    values: dict[str, str] = {}
    if secret_file:
        path = Path(secret_file)
        if path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line or line.startswith("#"):
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        if os.getenv(key):
            values[key] = os.environ[key]
    if values.get("base_url") and not values.get("OPENAI_BASE_URL"):
        values["OPENAI_BASE_URL"] = values["base_url"]
    if values.get("api_key") and not values.get("OPENAI_API_KEY"):
        values["OPENAI_API_KEY"] = values["api_key"]
    return values


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o-mini", timeout: float = 12.0):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.last_usage: dict[str, int] = {}
        self.last_retry_count = 0
        self.last_generation: dict[str, Any] = {}

    def complete(self, system: str, user: str, purpose: str = "general", generation_config: dict[str, Any] | None = None) -> str:
        config = resolve_generation_config(purpose, generation_config)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": config.temperature,
            "max_tokens": config.max_new_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        if usage:
            self.last_usage = {k: int(getattr(usage, k)) for k in ("prompt_tokens", "completion_tokens", "total_tokens") if getattr(usage, k, None) is not None}
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        self.last_generation = {"purpose": purpose, "max_new_tokens": config.max_new_tokens, "finish_reason": finish_reason, "truncated": finish_reason == "length", "eos_reached": finish_reason in {"stop", "eos"}}
        return choice.message.content or ""
