from __future__ import annotations

import json
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

