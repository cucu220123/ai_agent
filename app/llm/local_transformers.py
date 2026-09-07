from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.llm.generation import resolve_generation_config


class LocalTransformersLLM:
    """Lazy local Transformers adapter; optional and never imported in mock mode."""

    _MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = Path(model_path).name
        self._model = None
        self._tokenizer = None
        self.last_usage: dict[str, int] = {}
        self.last_retry_count = 0
        self.last_generation: dict[str, Any] = {}

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = os.getenv("LOCAL_LLM_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        cache_key = (self.model_path, device)
        if cache_key in self._MODEL_CACHE:
            self._tokenizer, self._model = self._MODEL_CACHE[cache_key]
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=False, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map={"": device},
            trust_remote_code=False,
            local_files_only=True,
        )
        self._MODEL_CACHE[cache_key] = (self._tokenizer, self._model)

    def complete(self, system: str, user: str, purpose: str = "general", generation_config: dict[str, Any] | None = None) -> str:
        self._load()
        config = resolve_generation_config(purpose, generation_config)
        max_chars = int(os.getenv("LOCAL_LLM_MAX_INPUT_CHARS", "60000"))
        char_truncated = len(user) > max_chars
        if len(user) > max_chars:
            user = user[:max_chars] + "\n[context truncated by local LLM adapter]"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"System: {system}\nUser: {user}\nAssistant:"
        context_length = int(getattr(self._model.config, "max_position_embeddings", 32768) or 32768)
        max_input_tokens = max(256, context_length - config.max_new_tokens)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(self._model.device)
        token_input_truncated = int(inputs.input_ids.shape[1]) >= max_input_tokens
        generate_kwargs: dict[str, Any] = {"max_new_tokens": config.max_new_tokens, "do_sample": config.temperature > 0}
        if config.temperature > 0:
            generate_kwargs["temperature"] = config.temperature
        import torch
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **generate_kwargs)
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        self.last_usage = {"prompt_tokens": int(inputs.input_ids.shape[1]), "completion_tokens": int(generated_tokens.shape[0]), "total_tokens": int(outputs.shape[1])}
        eos_id = self._tokenizer.eos_token_id
        eos_reached = bool(eos_id is not None and generated_tokens.numel() and int(generated_tokens[-1]) == int(eos_id))
        truncated = int(generated_tokens.shape[0]) >= config.max_new_tokens and not eos_reached
        self.last_generation = {"purpose": purpose, "max_new_tokens": config.max_new_tokens, "context_length": context_length, "finish_reason": "length" if truncated else "eos", "truncated": truncated, "eos_reached": eos_reached, "input_truncated": len(user) > max_chars or token_input_truncated}
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
