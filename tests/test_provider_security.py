import json

import pytest

from app.llm.security import load_credentials, sanitize
from app.llm.telemetry import TracedLLM


def test_inline_secret_is_data_and_never_part_of_url(tmp_path, monkeypatch):
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "secret.txt"
    key = "test-private-credential-12345"
    path.write_text(f'base_url="https://example.invalid/v1", api_key="{key}"\nexport OPENAI_MODEL="test-model"\n')
    values = load_credentials(path)
    assert values["OPENAI_BASE_URL"] == "https://example.invalid/v1"
    assert values["OPENAI_API_KEY"] == key
    assert key not in sanitize(f"failure {key} https://example.invalid/v1")
    assert "example.invalid" not in sanitize("https://example.invalid/v1")


def test_json_secret_and_env_precedence(tmp_path, monkeypatch):
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"base_url": "http://localhost:9000/v1", "api_key": "local-value", "model": "old"}))
    monkeypatch.setenv("OPENAI_MODEL", "new")
    assert load_credentials(path)["OPENAI_MODEL"] == "new"


def test_telemetry_records_failure_without_secret():
    class BadProvider:
        model = "test"
        def complete(self, *args, **kwargs):
            raise ValueError("Bearer abcdef123456 https://private.invalid/v1")
    llm = TracedLLM(BadProvider(), "fixture")
    with pytest.raises(RuntimeError):
        llm.complete("system", "user", purpose="planning")
    record = llm.calls[0]
    assert record["status"] == "error" and record["latency_ms"] >= 0
    assert "abcdef123456" not in str(record)
    assert "private.invalid" not in str(record)


def test_no_real_provider_does_not_silently_become_mock():
    from app.config import Settings
    from app.llm.factory import build_llm
    with pytest.raises(ValueError, match="No real LLM"):
        build_llm(Settings(llm_provider="auto", secret_file=None, local_instruction_model_path=None, local_model_path=None, openai_base_url=None, openai_api_key=None))

