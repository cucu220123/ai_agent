from __future__ import annotations

import json
import os
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import CapabilitySpec


class SemanticRetriever:
    """Dependency-light semantic retrieval using TF-IDF; embeddings can be swapped later."""

    def retrieve(self, spec: CapabilitySpec, documents: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        if not documents:
            return []
        texts = [json.dumps(item, ensure_ascii=False) for item in documents]
        query = json.dumps(spec.to_dict(), ensure_ascii=False)
        try:
            matrix = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1).fit_transform([query, *texts])
            scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()
        except ValueError:
            scores = [0.0] * len(documents)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)[:limit]
        return [{"score": round(float(score), 6), "source": item} for score, item in ranked]


class EmbeddingRetriever:
    """Local Transformers mean-pooling retriever; no network or sentence-transformers required."""

    _MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}

    def __init__(self, model_path: str, device: str | None = None):
        self.model_path = model_path
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        cache_key = (self.model_path, self.device)
        if cache_key in self._MODEL_CACHE:
            self._tokenizer, self._model = self._MODEL_CACHE[cache_key]
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        self._model.eval()
        self._MODEL_CACHE[cache_key] = (self._tokenizer, self._model)

    def retrieve(self, spec: CapabilitySpec, documents: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        if not documents:
            return []
        import torch
        import torch.nn.functional as F

        self._load()
        texts = [json.dumps(spec.to_dict(), ensure_ascii=False), *[json.dumps(item, ensure_ascii=False) for item in documents]]
        vectors = []
        batch_size = 16
        with torch.inference_mode():
            for index in range(0, len(texts), batch_size):
                encoded = self._tokenizer(texts[index : index + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                output = self._model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vectors.append(F.normalize(pooled, p=2, dim=1).cpu())
        matrix = torch.cat(vectors)
        scores = (matrix[1:] @ matrix[0]).tolist()
        ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)[:limit]
        return [{"score": round(float(score), 6), "source": item, "backend": "embedding", "model": self.model_path} for score, item in ranked]


class HybridSemanticRetriever:
    def __init__(self, embedding_model_path: str | None = None):
        self.tfidf = SemanticRetriever()
        self.embedding = EmbeddingRetriever(embedding_model_path) if embedding_model_path else None
        self.last_backend = "tfidf"
        self.last_error: str | None = None

    def retrieve(self, spec: CapabilitySpec, documents: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        if self.embedding:
            try:
                result = self.embedding.retrieve(spec, documents, limit)
                self.last_backend = "embedding"
                return result
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"[:500]
        self.last_backend = "tfidf"
        return [{**item, "backend": "tfidf"} for item in self.tfidf.retrieve(spec, documents, limit)]
