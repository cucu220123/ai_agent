# 正式真实模型证据

正式验收使用本地 OpenAI-compatible API：Qwen2.5-14B-Instruct 负责需求、抽取、规划、诊断和解释；Qwen3-Coder-30B-A3B-Instruct 负责代码生成与修复。实际调用的 provider、model、latency、usage 和状态保存在原始报告中。

| 证据 | 来源与范围 |
|---|---|
| [四阶段验收](FINAL_ACCEPTANCE.md) | 客户流失、相似任务、自修复、早期文本流程；58 条调用记录含失败与重试 |
| [公开文本独立测试](TEXT_ACCEPTANCE.md) | 7 次真实 API 调用；745 条最终测试，Accuracy/F1 约 0.824 |
| [知识抽取](../examples/acceptance_real_20260907/extracted_knowledge.json) | 文档/Python 语义与来源；age → Feature、churn → Target、ROC-AUC → Metric |
| [跨任务闭环](../examples/acceptance_real_20260907/closed_loop_proof.json) | 真实 Workflow → Curator → 第二次 Retrieval → Planner |
| [自修复](../examples/acceptance_real_20260907/self_repair_demo/) | 注入接口故障后，真实模型修复及重新执行；before/after/proof 保留 |

Mock 仅用于显式离线软件测试，不能作为模型能力证据。早期 benchmark、模板 fallback 和受控 prior 注入仅说明当时的开发实验，不替代上述正式验收。当前模型配置与历史 benchmark 的范围见 [MODEL_SELECTION](MODEL_SELECTION.md)。
