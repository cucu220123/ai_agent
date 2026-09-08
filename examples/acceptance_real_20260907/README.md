# 真实验收证据导航

完整分析见 [FINAL_ACCEPTANCE](../../docs/FINAL_ACCEPTANCE.md)。四阶段均使用真实本地 Qwen API，没有 Mock winner。公开文本独立测试另见 [TEXT_ACCEPTANCE](../../docs/TEXT_ACCEPTANCE.md)。

| 文件 | 用途 |
|---|---|
| `first.json` | 1200 行客户流失主任务；ROC-AUC 0.928877；冷启动没有实测历史 |
| `second.json` | 1050 行新数据任务；ROC-AUC 0.865567；检索并使用 first run |
| `closed_loop_proof.json` | 第一次/第二次 planning、search 与精确 LLM Planner context |
| `graph_retrieval_example.json` | 实际 Query→Entity→Graph paths→serialized evidence 的展示摘录 |
| `repair.json` | 明确接口故障注入，真实 LLM 修复后的 PASS |
| `self_repair_demo/before.py`、`after.py`、`proof.json` | 源码前后、原错误、诊断、修复调用、两次验证 |
| `self_repair_demo/next_retrieval.json` | 新失败经验被下一次真实图/案例检索找回 |
| `cross.json` | 16 条文本，三个配置实际执行；accuracy 0.50 / weighted F1 0.333，不能称为高质量分类器 |
| `report_extraction.json` | first 实测实验报告再次经真实 LLM 抽取，并保留来源 |
| `extracted_knowledge.json` | 正式材料抽取输出；age 为 Feature、churn 为 Target、ROC-AUC 为 Metric，包含 trace/provenance |
| `graph_after_first.graphml`、`knowledge_snapshot.graphml` | 第一次写回与最终图快照 |
| `workspace/generated/` | 每个 candidate 的生成尝试、repair JSON、不可变 vN.py、meta |
| `workspace/reports/` | 原始运行报告；包括失败尝试，不隐藏失败 |
| `report_revisions/` | 算法已通过、解释失败后的原报告与解释单独恢复；原始 hash 和测量字段保持 |
| `explanation_reviews/` | 后续解释复核，绑定原报告内容 hash；rejected 中保留被新检查拒绝的旧 review |
| `execution_logs/` | 脱敏的主任务、闭环、修复、跨任务、解释复核及失败过程日志 |
| `environment.json`、`backend_transition.json`、`serving_32k_probe.json` | 实际依赖、模型和服务变更；初始 NF4 健康文件不是最终 vLLM 配置 |
| `manifest.json` | 四阶段、58 条调用记录及当时源码快照；不是所有调用都成功 |
| `verifier.json` | 只读核验 4 runs / 21 source versions 的结果 |
| `clone_verification.json` | Git 导出的全新目录再次核验，证明已提交产物完整 |
| `pytest.log`、`test-results.xml` | 最后一次完整自动测试统计 |
| `ui_checks.json` | 实际 UI 人工检查范围 |
| `credential_scan.json` | 凭证扫描范围与结果，不保存凭证内容 |
| `service_cleanup.json` | 验收结束后临时 GPU 服务停服检查；报告界面保留 |

## 无需模型复查已保存结果

仓库根目录安装应用依赖后运行：

```bash
python scripts/verify_evidence.py --output examples/acceptance_real_20260907
LLM_PROVIDER=mock ENABLE_LOCAL_EMBEDDING=0 pytest -q
```

verifier 从历史绝对路径中的 `/workspace/` 后缀定位当前 clone 内的源码，核验每个版本 SHA256；不会执行旧源码或重写报告。SQLite 缓存按项目策略未提交；GraphML 和提取结果用于审阅。重新进行学习实验会建立新的数据库。

## 重新运行真实模型实验

按 [README 的真实 API 配置](../../README.md#3-真实-llm-配置) 启动/配置实际服务，然后使用新的 output 目录，避免混淆本次证据：

```bash
python scripts/run_acceptance.py --provider openai --output examples/my_acceptance
python scripts/review_explanations.py --provider openai --output examples/my_acceptance
python scripts/verify_evidence.py --output examples/my_acceptance
```

运行可按 first/second/repair/cross/verify 阶段续跑。代码由真实模型生成，可能有候选失败或需要多轮修复，不保证每次运行相同。报告的 PASS 表示配置的检查已通过，未设置质量阈值的文本任务不能因此宣称业务质量合格。

源数据、实测版本和日志保留当时状态；`manifest.git_head_at_verification` 是核验时 Git HEAD，并非所有历史调用都运行于那个 commit。解释复核只修改单独 review；解释恢复有可检查的原报告修订谱系，均不改实际算法测量。
