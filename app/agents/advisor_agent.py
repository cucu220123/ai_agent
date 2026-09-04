from __future__ import annotations

import json
from typing import Any

from app.llm.contracts import extract_json_object
from app.models import CapabilitySpec, KnowledgeContext


class AdvisorAgent:
    """Optional structured LLM advisor. Its output is advisory and validated before use."""

    def __init__(self, llm: Any):
        self.llm = llm

    def run(self, spec: CapabilitySpec, knowledge: KnowledgeContext) -> dict[str, Any]:
        prompt = {
            "task": "从能力描述和检索知识中给出算法工厂建议",
            "required_json": {"rationale": "string", "candidate_algorithms": ["logistic_regression"], "metric_thresholds": {"roc_auc": 0.75}, "risks": ["string"]},
            "spec": spec.to_dict(),
            "knowledge": knowledge.to_dict(),
        }
        raw = self.llm.complete("你是严格的算法方案顾问。只输出 JSON 对象，不要 Markdown。", json.dumps(prompt, ensure_ascii=False))
        parsed = extract_json_object(raw)
        return parsed or {"rationale": raw[:1000], "candidate_algorithms": [], "metric_thresholds": {}, "risks": ["LLM 返回不是结构化 JSON"]}

