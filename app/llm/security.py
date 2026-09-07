"""Parse credential configuration as data and redact before crossing log boundaries."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_SENSITIVE: set[str] = set()
ALIASES = {"base_url": "OPENAI_BASE_URL", "api_base": "OPENAI_BASE_URL", "api_key": "OPENAI_API_KEY", "key": "OPENAI_API_KEY", "model": "OPENAI_MODEL"}


def register_secret(value: str | None) -> None:
    if value and len(value) >= 4:
        _SENSITIVE.add(value)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if any(part in str(k).lower() for part in ("api_key", "password", "authorization")) else sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if not isinstance(value, str):
        return value
    for secret in sorted(_SENSITIVE, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", value)
    value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED]", value)
    value = re.sub(r"https?://[^\s\"'<>]+", lambda m: "[endpoint:" + hashlib.sha256(m[0].encode()).hexdigest()[:10] + "]", value)
    return value


def load_credentials(secret_file: str | Path | None = None) -> dict[str, str]:
    values: dict[str, str] = {}
    if secret_file and Path(secret_file).is_file():
        content = Path(secret_file).read_text(encoding="utf-8-sig")
        if content.lstrip().startswith("{"):
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("secret configuration must be a JSON object")
            values = {str(k): str(v) for k, v in parsed.items() if isinstance(v, (str, int))}
        else:
            # Supports dotenv and Python SDK inline keyword syntax without executing it.
            pattern = r"(?:^|[,\s(])(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^,\n#)]+)"
            for match in re.finditer(pattern, content):
                key, raw = match.groups()
                raw = raw.strip()
                values[key] = str(ast.literal_eval(raw)) if raw.startswith(('"', "'")) else raw
    for alias, canonical in ALIASES.items():
        if alias in values and canonical not in values:
            values[canonical] = values[alias]
    for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        if os.getenv(key):
            values[key] = os.environ[key]
    register_secret(values.get("OPENAI_API_KEY"))
    endpoint = values.get("OPENAI_BASE_URL")
    if endpoint:
        parts = urlsplit(endpoint)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment or re.search(r"[\s\"',]", endpoint):
            raise ValueError("invalid OpenAI-compatible base URL; inspect configuration format")
        values["OPENAI_BASE_URL"] = endpoint.rstrip("/")
    return {k: v for k, v in values.items() if k in {"OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"}}
