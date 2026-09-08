# Historical evidence — 非正式验收

| 文件 | 历史范围 |
|---|---|
| `real_knowledge_extraction.json` | 早期抽取实验，存在实体角色类型错误；保留原始字节供审计，不代表当前 schema |
| `controlled_prior_injection.json` | 早期手工注入 prior 的输出，含构造的 0.99 指标；没有真实训练或 ValidationRunner 测量 |

正式抽取以 [extracted_knowledge.json](../../../examples/acceptance_real_20260907/extracted_knowledge.json) 为准：age → Feature、churn → Target、ROC-AUC → Metric。

正式跨任务闭环仅引用 [closed_loop_proof.json](../../../examples/acceptance_real_20260907/closed_loop_proof.json)，来自真实 Workflow → Curator → 第二次 Retrieval/Planner。受控注入不能用作该闭环的实测证据。
