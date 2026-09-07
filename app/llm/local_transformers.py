from __future__ import annotations

import os
from pathlib import Path


class LocalTransformersLLM:
    """Lazy local Transformers adapter; optional and never imported in mock mode."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = Path(model_path).name
        self._model = None
        self._tokenizer = None
        self.last_usage: dict[str, int] = {}
        self.last_retry_count = 0

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    def complete(self, system: str, user: str) -> str:
        self._load()
        max_chars = int(os.getenv("LOCAL_LLM_MAX_INPUT_CHARS", "14000"))
        if len(user) > max_chars:
            user = user[:max_chars] + "\n[context truncated by local LLM adapter]"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"System: {system}\nUser: {user}\nAssistant:"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=384, do_sample=False)
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        self.last_usage = {"prompt_tokens": int(inputs.input_ids.shape[1]), "completion_tokens": int(generated_tokens.shape[0]), "total_tokens": int(outputs.shape[1])}
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
