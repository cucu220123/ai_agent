from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models import CapabilitySpec
from app.retrieval.semantic import EmbeddingRetriever, SemanticRetriever


DOCUMENTS = [
    {"id": "churn_case", "text": "客户流失预测，表格二分类，类别不平衡，使用逻辑回归和 ROC-AUC 评估"},
    {"id": "inventory_case", "text": "库存需求回归预测，使用随机森林回归并评价 MAE"},
    {"id": "text_case", "text": "文本情感分类，TF-IDF 和 macro F1"},
    {"id": "anomaly_case", "text": "设备传感器异常检测，Isolation Forest"},
]


def evaluate(results):
    ranking = [item["source"]["id"] for item in results]
    rank = ranking.index("churn_case") + 1 if "churn_case" in ranking else 999
    return {"ranking": ranking, "relevant_rank": rank, "reciprocal_rank": 1.0 / rank}


def main() -> None:
    spec = CapabilitySpec(raw_description="客户流失预测，表格数据类别不平衡，要求 ROC-AUC 大于 0.8", domain="customer_churn", capability_name="客户流失预测", task_type="binary_classification", data_type="tabular", target_column="churn", metrics=["roc_auc"], class_imbalance={"is_imbalanced": True})
    started = time.perf_counter()
    tfidf = SemanticRetriever().retrieve(spec, DOCUMENTS, 4)
    tfidf_latency = time.perf_counter() - started
    model_path = "/data/public_checkpoints/huggingface_models/shibing624-text2vec-base-chinese"
    started = time.perf_counter()
    embedding = EmbeddingRetriever(model_path).retrieve(spec, DOCUMENTS, 4)
    embedding_latency = time.perf_counter() - started
    payload = {"query": spec.raw_description, "tfidf": {**evaluate(tfidf), "latency_seconds": round(tfidf_latency, 4), "scores": tfidf}, "embedding": {**evaluate(embedding), "latency_seconds": round(embedding_latency, 4), "model": model_path, "scores": embedding}}
    output = PROJECT_ROOT / "docs/evidence/retrieval_backend_comparison.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

