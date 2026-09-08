"""Optional diagnostic for an explicitly configured OpenAI-compatible endpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.llm.security import load_credentials


def probe(secret_file: str | Path) -> dict:
    from openai import OpenAI
    values = load_credentials(secret_file)
    if not values.get("OPENAI_BASE_URL") or not values.get("OPENAI_API_KEY"):
        raise ValueError("configuration lacks an endpoint or API key")
    client = OpenAI(base_url=values["OPENAI_BASE_URL"], api_key=values["OPENAI_API_KEY"], timeout=25, max_retries=0)
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "provider": "configured_openai_compatible", "endpoint_fingerprint": hashlib.sha256(values["OPENAI_BASE_URL"].encode()).hexdigest()[:12], "configuration_parsed": True, "attempts": []}
    operations = [("models.list", None), *(('chat.completions', m) for m in dict.fromkeys([values.get("OPENAI_MODEL", "gpt-4o-mini"), "gpt-4o"]))]
    for operation, model in operations:
        record = {"operation": operation, "model": model}
        try:
            if model:
                response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Return only OK."}], max_tokens=8)
                record.update(status="ok", usage=response.usage.model_dump() if response.usage else {})
            else:
                models = client.models.list()
                record.update(status="ok", model_count=len(models.data))
        except Exception as exc:
            message = str(exc).lower()
            reason = "quota_exhausted" if any(token in message for token in ("quota", "额度", "insufficient_quota")) else "authentication_failed" if getattr(exc, "status_code", None) == 401 else "provider_error"
            record.update(status="error", exception_type=type(exc).__name__, http_status=getattr(exc, "status_code", None), reason=reason)
        result["attempts"].append(record)
    result["chat_success"] = any(a["status"] == "ok" and a["operation"] == "chat.completions" for a in result["attempts"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = probe(args.secret_file)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
