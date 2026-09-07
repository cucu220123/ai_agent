from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


LLMPurpose = Literal["requirement", "planning", "extraction", "code_generation", "repair", "critique", "explanation", "general"]


@dataclass(frozen=True)
class GenerationConfig:
    purpose: LLMPurpose = "general"
    max_new_tokens: int = 768
    temperature: float = 0.0
    json_mode: bool = False


DEFAULT_GENERATION_CONFIGS: dict[str, GenerationConfig] = {
    "requirement": GenerationConfig("requirement", 1024, 0.0, True),
    "planning": GenerationConfig("planning", 1280, 0.0, True),
    "extraction": GenerationConfig("extraction", 1792, 0.0, True),
    "code_generation": GenerationConfig("code_generation", 3072, 0.0, False),
    "repair": GenerationConfig("repair", 3072, 0.0, False),
    "critique": GenerationConfig("critique", 1024, 0.0, True),
    "explanation": GenerationConfig("explanation", 1024, 0.0, True),
    "general": GenerationConfig(),
}


def resolve_generation_config(purpose: str = "general", overrides: dict[str, Any] | None = None) -> GenerationConfig:
    base = DEFAULT_GENERATION_CONFIGS.get(purpose, DEFAULT_GENERATION_CONFIGS["general"])
    values = {"purpose": base.purpose, "max_new_tokens": base.max_new_tokens, "temperature": base.temperature, "json_mode": base.json_mode}
    values.update(overrides or {})
    return GenerationConfig(**values)

