from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    # python-dotenv is optional for the offline path.
    pass

DEFAULT_SECRET_FILE = Path("/data/xiaotianqi/gen_eval/eval/secret.txt")
DEFAULT_LOCAL_MODEL = Path("/data/public_checkpoints/huggingface_models/Qwen2.5-1.5B-Instruct")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    generated_dir: Path = PROJECT_ROOT / "generated"
    reports_dir: Path = PROJECT_ROOT / "reports"
    knowledge_db: Path = PROJECT_ROOT / "knowledge.sqlite"
    graphml_path: Path = PROJECT_ROOT / "knowledge.graphml"
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto").lower()
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    secret_file: Path | None = Path(os.environ["AI_FACTORY_SECRET_FILE"]) if os.getenv("AI_FACTORY_SECRET_FILE") else (DEFAULT_SECRET_FILE if DEFAULT_SECRET_FILE.exists() else None)
    local_model_path: str | None = os.getenv("LOCAL_MODEL_PATH", str(DEFAULT_LOCAL_MODEL) if DEFAULT_LOCAL_MODEL.exists() else "") or None
    validation_timeout_seconds: int = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "90"))
    max_repair_rounds: int = int(os.getenv("MAX_REPAIR_ROUNDS", "3"))

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.generated_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
