# 公开文本与独立最终测试：683902e2a7cf

详细结果、复现命令与限制见 [TEXT_ACCEPTANCE.md](../../docs/TEXT_ACCEPTANCE.md)。

| 证据 | 含义 |
|---|---|
| `evaluation_policy.json` | 运行前固定的切分 hash、阈值、候选/修复预算 |
| `development.json` | 真实 LLM 理解、GraphRAG、规划、三个代码候选、开发验证、写回与解释 |
| `initial_knowledge.graphml`、`knowledge_initialization.json` | 上一轮实测知识与失败经验的起点 |
| `knowledge_after_development.graphml` | 当前三个候选写回后的图 |
| `extracted_knowledge.json` | 继承的真实抽取与来源信息，本轮为缓存复用 |
| `workspace/generated/` | 三个真实模型生成的不可变 v1.py、原输出及 metadata |
| `final_evaluation/commitment.json` | 最终测量前固定 winner/代码/数据/报告 hash |
| `final_evaluation/frozen_winner.py` | 最终评估的确切源码 |
| `final_evaluation/result.json` | 745 条独立测试，accuracy 0.824161、weighted F1 0.824138 |
| `verifier.json` | 三个版本、七次真实调用及两阶段完整性核验 |
| `execution.log`、`pytest.log`、`test-results.xml` | 脱敏执行日志与 86 项通过的完整测试 |
| `ui_checks.json` | 实际浏览器检查，分别显示开发与最终测量 |
| `credential_scan.json` | 工作区及新增 Git 历史的凭证扫描 |

原始公共数据、归属与固定切分在 [data/uci_sentiment](../../data/uci_sentiment/)。严格区分开发解释和最终评估；最终标签、预测与分数未送给规划/修复 Agent。本轮不用 Mock，也没有修改旧 16 条 smoke demo 的报告。
