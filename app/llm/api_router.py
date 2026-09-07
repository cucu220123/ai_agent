"""Optional purpose-based API routing with truthful per-call model telemetry."""
from typing import Any


class APIRoleRouter:
    def __init__(self, instruction: Any, coder: Any):
        self.instruction, self.coder = instruction, coder
        self.model = instruction.model
        self.last_provider = instruction.last_provider
        self.last_usage, self.last_generation = {}, {}
        self.last_retry_count = 0

    def complete(self, system: str, user: str, purpose: str = "general", generation_config=None) -> str:
        backend = self.coder if purpose in {"code_generation", "repair"} else self.instruction
        try:
            return backend.complete(system, user, purpose=purpose, generation_config=generation_config)
        finally:
            self.model = backend.model
            self.last_provider = backend.last_provider
            self.last_usage = backend.last_usage
            self.last_generation = backend.last_generation
            self.last_retry_count = backend.last_retry_count
