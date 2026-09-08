from __future__ import annotations

import os
from dataclasses import dataclass, replace, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    # python-dotenv is optional for the offline path.
    pass

DEFAULT_LOCAL_MODEL = Path("/data/public_checkpoints/huggingface_models/Qwen2.5-14B-Instruct")
DEFAULT_CODER_MODEL = Path("/data/public_checkpoints/huggingface_models/Qwen2.5-Coder-3B-Instruct")
DEFAULT_EMBEDDING_MODEL = Path("/data/public_checkpoints/huggingface_models/shibing624-text2vec-base-chinese")


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
    openai_api_key: str | None = field(default=None, repr=False)
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_coder_base_url: str | None = os.getenv("OPENAI_CODER_BASE_URL")
    openai_coder_model: str | None = os.getenv("OPENAI_CODER_MODEL")
    openai_coder_api_key: str | None = field(default=None, repr=False)
    secret_file: Path | None = Path(os.environ["AI_FACTORY_SECRET_FILE"]) if os.getenv("AI_FACTORY_SECRET_FILE") else None
    local_model_path: str | None = os.getenv("LOCAL_MODEL_PATH", str(DEFAULT_LOCAL_MODEL) if DEFAULT_LOCAL_MODEL.exists() else "") or None
    local_instruction_model_path: str | None = os.getenv("LOCAL_INSTRUCTION_MODEL_PATH", str(DEFAULT_LOCAL_MODEL) if DEFAULT_LOCAL_MODEL.exists() else "") or None
    local_coder_model_path: str | None = os.getenv("LOCAL_CODER_MODEL_PATH", str(DEFAULT_CODER_MODEL) if DEFAULT_CODER_MODEL.exists() else "") or None
    embedding_model_path: str | None = os.getenv("EMBEDDING_MODEL_PATH") or None
    validation_timeout_seconds: int = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "90"))
    validation_memory_mb: int = int(os.getenv("VALIDATION_MEMORY_MB", "16384"))
    max_repair_rounds: int = int(os.getenv("MAX_REPAIR_ROUNDS", "3"))
    planning_context_max_chars: int = int(os.getenv("PLANNING_CONTEXT_MAX_CHARS", "12000"))
    llm_code_candidate_budget: int = int(os.getenv("LLM_CODE_CANDIDATE_BUDGET", "2"))
    codegen_mode: str = os.getenv("CODEGEN_MODE", "free_form_llm")
    llm_timeout_seconds: float = 240.0
    strict_real_llm: bool = True
    beam_width: int = 3
    validation_seed_variance: float = 0.02
    validation_repeats: int = 3
    validation_cv_folds: int = 0
    bootstrap_knowledge: bool = True

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.generated_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings = replace(settings, llm_provider=os.getenv("LLM_PROVIDER", settings.llm_provider), openai_base_url=os.getenv("OPENAI_BASE_URL", settings.openai_base_url), openai_api_key=os.getenv("OPENAI_API_KEY"), openai_model=os.getenv("OPENAI_MODEL", settings.openai_model), llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "240")), beam_width=int(os.getenv("BEAM_WIDTH", "3")), strict_real_llm=os.getenv("STRICT_REAL_LLM", "1") == "1")
    if os.getenv("AI_FACTORY_WORKSPACE"):
        root = Path(os.environ["AI_FACTORY_WORKSPACE"]).resolve()
        settings = replace(settings, project_root=root, data_dir=root / "data", generated_dir=root / "generated", reports_dir=root / "reports", knowledge_db=root / "knowledge.sqlite", graphml_path=root / "knowledge.graphml")
    if settings.embedding_model_path is None and DEFAULT_EMBEDDING_MODEL.exists() and os.getenv("ENABLE_LOCAL_EMBEDDING", "1") == "1":
        settings = replace(settings, embedding_model_path=str(DEFAULT_EMBEDDING_MODEL))
    settings.ensure_dirs()
    return settings
