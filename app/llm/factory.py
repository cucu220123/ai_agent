from __future__ import annotations

from app.config import Settings
from app.llm.mock import MockLLM
from app.llm.openai_compatible import OpenAICompatibleLLM, load_openai_settings


def build_llm(settings: Settings):
    provider = settings.llm_provider
    if provider == "openai":
        values = load_openai_settings(settings.secret_file)
        base_url = values.get("OPENAI_BASE_URL") or settings.openai_base_url
        api_key = values.get("OPENAI_API_KEY") or settings.openai_api_key
        model = values.get("OPENAI_MODEL") or settings.openai_model
        if not base_url or not api_key:
            raise ValueError("openai provider requires OPENAI_BASE_URL and OPENAI_API_KEY")
        return OpenAICompatibleLLM(base_url, api_key, model)
    if provider == "local":
        if not settings.local_model_path:
            raise ValueError("local provider requires LOCAL_MODEL_PATH")
        from app.llm.local_transformers import LocalTransformersLLM

        return LocalTransformersLLM(settings.local_model_path)
    return MockLLM()

