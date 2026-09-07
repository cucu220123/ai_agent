from __future__ import annotations

from app.config import Settings
from app.llm.mock import MockLLM
from app.llm.openai_compatible import OpenAICompatibleLLM, load_openai_settings


class AutoFallbackLLM:
    """Prefer configured cloud API, then local model, and expose the final provider in traces."""

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback
        self.last_provider = "primary"
        self.last_usage = {}
        self.last_retry_count = 0
        self.last_generation = {}
        self.model = getattr(primary, "model", None)
        self._primary_failed = False

    def complete(self, system: str, user: str, purpose: str = "general", generation_config=None) -> str:
        try:
            if self._primary_failed:
                raise RuntimeError("primary provider circuit breaker open")
            result = self.primary.complete(system, user, purpose=purpose, generation_config=generation_config)
            self.last_provider = "openai"
            self.last_usage = getattr(self.primary, "last_usage", {})
            self.last_generation = getattr(self.primary, "last_generation", {})
            return result
        except Exception as primary_error:
            self._primary_failed = True
            if self.fallback is None:
                raise primary_error
            result = self.fallback.complete(system, user, purpose=purpose, generation_config=generation_config)
            self.last_provider = "local_fallback"
            self.model = getattr(self.fallback, "model", None)
            self.last_usage = getattr(self.fallback, "last_usage", {})
            self.last_generation = getattr(self.fallback, "last_generation", {})
            return result

    def provider_label(self) -> str:
        return self.last_provider


def build_llm(settings: Settings):
    provider = settings.llm_provider
    if provider == "auto":
        values = load_openai_settings(settings.secret_file)
        base_url = values.get("OPENAI_BASE_URL") or settings.openai_base_url
        api_key = values.get("OPENAI_API_KEY") or settings.openai_api_key
        primary = OpenAICompatibleLLM(base_url, api_key, values.get("OPENAI_MODEL") or settings.openai_model, timeout=12.0) if base_url and api_key else None
        fallback = None
        instruction_path = settings.local_instruction_model_path or settings.local_model_path
        if instruction_path:
            from app.llm.router import LocalModelRouter
            fallback = LocalModelRouter(instruction_path, settings.local_coder_model_path)
        if primary:
            return AutoFallbackLLM(primary, fallback)
        if fallback:
            return fallback
        return MockLLM()
    if provider == "openai":
        values = load_openai_settings(settings.secret_file)
        base_url = values.get("OPENAI_BASE_URL") or settings.openai_base_url
        api_key = values.get("OPENAI_API_KEY") or settings.openai_api_key
        model = values.get("OPENAI_MODEL") or settings.openai_model
        if not base_url or not api_key:
            raise ValueError("openai provider requires OPENAI_BASE_URL and OPENAI_API_KEY")
        return OpenAICompatibleLLM(base_url, api_key, model)
    if provider == "local":
        instruction_path = settings.local_instruction_model_path or settings.local_model_path
        if not instruction_path:
            raise ValueError("local provider requires LOCAL_INSTRUCTION_MODEL_PATH or LOCAL_MODEL_PATH")
        from app.llm.router import LocalModelRouter
        return LocalModelRouter(instruction_path, settings.local_coder_model_path)
    return MockLLM()
