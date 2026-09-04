from __future__ import annotations


class LocalTransformersLLM:
    """Lazy local Transformers adapter; optional and never imported in mock mode."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

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
        prompt = f"System: {system}\nUser: {user}\nAssistant:"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

