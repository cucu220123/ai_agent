from __future__ import annotations

import os
from pathlib import Path


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
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=45, max_retries=1)

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""

