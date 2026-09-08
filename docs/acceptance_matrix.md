# 原题要求验收矩阵

正式结果以 [FINAL_ACCEPTANCE](FINAL_ACCEPTANCE.md) 和 [TEXT_ACCEPTANCE](TEXT_ACCEPTANCE.md) 为准。

| 要求 | 当前实现 | 证据或限制 |
|---|---|---|
| 真实 LLM | Qwen2.5-14B + Qwen3-Coder-30B，经本地 API | 正式报告有逐调用记录；Mock 仅用于离线测试 |
| 行业场景 | 客户流失预测、公开文本分类 | 合成客户数据与小型公开语料不能代表生产泛化 |
| 知识抽取与图谱 | LLM + Python AST、来源校验、SQLite/NetworkX | [正式抽取](../examples/acceptance_real_20260907/extracted_knowledge.json) |
| 需求到代码 | 结构化理解、GraphRAG、规划、搜索、真实代码生成 | 正常严格路径不接受 Mock/模板 winner |
| 检索与经验复用 | 1–3 hop 子图、向量/文档融合、相似实测案例 | [正式闭环](../examples/acceptance_real_20260907/closed_loop_proof.json) 来自 Workflow → Curator → 第二次 Retrieval |
| 搜索与多候选 | 有限算法/预处理/参数组合，实际执行保留候选 | 一轮 Beam 剪枝，不是 MCTS 或全局最优搜索 |
| 自动验证 | 接口、可信指标、稳定性、鲁棒性、资源与安全 | prototype sandbox，不是生产隔离 |
| 多轮修复与版本 | Critic/Repair、不可变源码、hash 和父版本 | 正式自修复 before/after/proof 保留 |
| 回写与解释 | 全候选/版本写回、失败经验、基于证据的解释 | 不能证明任意自由文本完全无幻觉 |
| CLI/API/Web | 提交任务、子图、候选、代码、验证、修复、调用轨迹 | 同步本机原型；FastAPI 自动接口文档 |
| 插件与跨任务 | 算法注册、renderer、自定义可信指标；回归/异常等测试 | 全新任务仍需专用协议与验证逻辑 |
| 文档与测试 | README、schema、架构、依赖、测试和封存报告 | 不把历史 benchmark 或 controlled prior injection 当正式验收 |

## 异常检测评价边界

有标签 anomaly detection 使用 F1、Precision、Recall；无标签任务可以执行并输出 `anomaly_score`、`anomaly_rate` 等观测统计。当前 `MetricRegistry` 对无 target 的 anomaly 使用 `runtime_seconds` 进行工程选择；runtime 是运行成本与资源指标，不能在缺少 ground truth 时解释为可靠的算法质量评价。尚未实现完整的 unsupervised quality proxy，属于 Future Work。

## 指标解释

客户流失主任务 ROC-AUC 约 0.9289，但 F1 约 0.3934、recall 约 0.261，且数据为合成数据。公开文本独立测试 Accuracy/F1 约 0.824；与旧 16 条 smoke demo 使用不同数据和协议，不能作同数据提升比较。
