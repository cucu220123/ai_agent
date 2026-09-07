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



def test_context_budget_recovery_preserves_input_and_records_effective_tokens():
    from types import SimpleNamespace
    from app.llm.openai_compatible import OpenAICompatibleLLM
    calls = []
    class ContextError(Exception):
        status_code = 400
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ContextError('You passed 9985 input tokens and requested 6400 output tokens. However, the model\'s context length is only 16384 tokens.')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'), finish_reason='stop')], usage=None)
    llm = OpenAICompatibleLLM('http://127.0.0.1:1/v1', 'local-fixture', 'fixture')
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert llm.complete('full system', 'all requirement and evidence', purpose='repair') == '{}'
    assert calls[0]['messages'] == calls[1]['messages']
    assert calls[1]['max_tokens'] == 6271
    assert llm.last_retry_count == 1
    assert llm.last_generation['budget_adjustments'] == [{'requested': 6400, 'effective': 6271, 'reason': 'provider_reported_context_limit'}]
    assert llm._context_output_budget('passed 16400 input tokens; context length is only 16384 tokens', 6400) is None
    assert llm._context_output_budget('unrelated bad request', 6400) is None
