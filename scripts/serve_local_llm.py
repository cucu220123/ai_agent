"""Optional loopback-only OpenAI-compatible API backed by real local weights.

No cloud key is needed. Single serialized inference queue; never expose publicly.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.llm.local_transformers import LocalTransformersLLM

app = FastAPI(title="Local real model API")
model: LocalTransformersLLM | None = None
lock = threading.Lock()


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = Field(default=2400, ge=1, le=8000)
    temperature: float = Field(default=0, ge=0, le=2)
    response_format: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": model.model if model else None, "real_weights": True, "quantization": "nf4" if model and getattr(model._model, "is_loaded_in_4bit", False) else "native"}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": model.model, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def complete(request: CompletionRequest) -> dict:
    if request.model != model.model:
        raise HTTPException(404, "model is not loaded")
    if request.response_format and request.response_format.get("type") == "json_schema":
        raise HTTPException(400, "response_format json_schema unsupported; use json_object plus application validation")
    system = "\n".join(m["content"] for m in request.messages if m["role"] == "system")
    user = "\n".join(m["content"] for m in request.messages if m["role"] != "system")
    json_mode = bool(request.response_format)
    if json_mode:
        system += "\nReturn exactly one valid JSON object, no markdown."
    with lock:
        import torch
        with torch.inference_mode():
            output = model.complete(system, user, generation_config={"max_new_tokens": request.max_tokens, "temperature": request.temperature, "json_mode": json_mode})
        usage = dict(model.last_usage)
        truncated = model.last_generation.get("truncated", False)
    return {"id": "local-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()), "model": model.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "length" if truncated else "stop"}], "usage": usage}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18086)
    args = parser.parse_args()
    model = LocalTransformersLLM(args.model)
    model._load()
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)

