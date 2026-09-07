from __future__ import annotations


class MockLLM:
    """Offline provider used for reproducible demos and tests."""

    def complete(self, system: str, user: str, purpose: str = "general", generation_config=None) -> str:
        return "离线 Mock LLM：已根据知识库和验证反馈采用受约束的算法模板。"
