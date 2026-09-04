from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.workflow import AlgorithmFactoryWorkflow


app = FastAPI(title="AI Algorithm Factory", version="0.1.0")


class RunRequest(BaseModel):
    description: str = Field(min_length=5)
    data_path: str
    provider: Literal["mock", "openai", "local"] = "mock"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai-algorithm-factory"}


@app.post("/run")
def run(request: RunRequest) -> dict:
    try:
        from app.config import Settings, get_settings

        base = get_settings()
        settings = Settings(**{**base.__dict__, "llm_provider": request.provider})
        result = AlgorithmFactoryWorkflow(settings).run(request.description, request.data_path)
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"workflow failed: {type(exc).__name__}: {exc}") from exc

