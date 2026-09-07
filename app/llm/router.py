from __future__ import annotations

from typing import Any

from app.llm.local_transformers import LocalTransformersLLM


class LocalModelRouter:
    """Route language understanding and coding tasks to separately benchmarked models."""

    CODING_PURPOSES = {"code_generation", "repair"}

    def __init__(self, instruction_model_path: str, coder_model_path: str | None = None):
        self.instruction = LocalTransformersLLM(instruction_model_path)
        self.coder = LocalTransformersLLM(coder_model_path or instruction_model_path)
        self.model = self.instruction.model
        self.last_provider = "local_instruction"
        self.last_usage: dict[str, int] = {}
        self.last_generation: dict[str, Any] = {}
        self.last_retry_count = 0

    def complete(self, system: str, user: str, purpose: str = "general", generation_config=None) -> str:
        provider = self.coder if purpose in self.CODING_PURPOSES else self.instruction
        self.last_provider = "local_coder" if purpose in self.CODING_PURPOSES else "local_instruction"
        result = provider.complete(system, user, purpose=purpose, generation_config=generation_config)
        self.model = provider.model
        self.last_usage = provider.last_usage
        self.last_generation = provider.last_generation
        return result

