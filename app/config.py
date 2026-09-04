from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    generated_dir: Path = PROJECT_ROOT / "generated"
    reports_dir: Path = PROJECT_ROOT / "reports"
    knowledge_db: Path = PROJECT_ROOT / "knowledge.sqlite"
    graphml_path: Path = PROJECT_ROOT / "knowledge.graphml"
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").lower()
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    secret_file: Path | None = Path(os.environ["AI_FACTORY_SECRET_FILE"]) if os.getenv("AI_FACTORY_SECRET_FILE") else None
    local_model_path: str | None = os.getenv("LOCAL_MODEL_PATH")
    validation_timeout_seconds: int = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "90"))
    max_repair_rounds: int = int(os.getenv("MAX_REPAIR_ROUNDS", "3"))

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.generated_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings

