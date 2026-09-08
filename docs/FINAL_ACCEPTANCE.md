# 系统验收报告

本报告记录客户流失、相似任务、自修复和文本流程四个阶段的验收结果。公开文本分类与独立最终测试见 [TEXT_ACCEPTANCE.md](TEXT_ACCEPTANCE.md)。各报告的测试统计对应其验收版本。

本页对应 `examples/acceptance_real_20260907/` 中的原始证据。运行跨越 2026-09-07 UTC 至次日北京时间。证据中的时间使用 UTC；算法版本 hash 绑定实际执行的代码，不把历史运行改标为最新 commit 的运行。

## 基线审计与实现改进

基线为 `7dea7e6` 加服务器已有未提交工作；原有工作已备份、保留。实际 baseline pytest 有 **6 个收集错误**，demo 因 generator 缩进 SyntaxError 失败。更早 README 的通过统计不能代表这个 checkout。完整审计见 [TECHNICAL_AUDIT.md](TECHNICAL_AUDIT.md)。

| 改造前实际缺口 | 现在进入执行路径的实现 |
|---|---|
| API 失败或代码预算耗尽后，模板可能支撑最终 PASS | 默认严格真实模式；实际 provider、模型、每次调用和失败均记录，真实验收禁止 Mock/模板获胜 |
| 结构化理解可覆盖真实数据事实，规则候选残留 | LLM JSON + Pydantic + 语义恢复；可信 CSV profile 与模型推断分离 |
| 抽取需单独运行，来源跨度未核对 | bootstrap 从 Markdown/Python AST 材料调用真实 LLM；实验 JSON 再抽取；实体、关系与原文跨度/hash 校验 |
| 图虽有遍历，但链接范围宽、路径与规划依据不完整 | 任务兼容的 1–3 hop NetworkX 搜索、路径分数、SubgraphSerializer、语义文档融合；精确 Planner context 保存 |
| 只写 winner；失败版本和成功替代方案丢失 | 全候选、每轮 ValidationRun/Version/Failure/Repair 写回，实际 Task B 检索 Task A |
| LLM 规划只影响算法名称 | 合法参数、预处理建议、依据 ID 进入候选；算法×预处理×配置实际扩展与剪枝 |
| 生成 evaluate 可自报指标；资源与稳定性部分只展示 | 可信父进程重算指标并比对；执行方差、RSS/CPU/延迟限制、robustness 与 protocol 检查 |
| 继承环境凭证，AST 可绕过 I/O；无不可变版本 | 清理子进程环境、临时目录、资源限额、audit、可用时 network namespace；v1→v2 源码/hash/父版本 |
| Web 偏 JSON 展示，历史 demo 注入手工分数 | 相关子图与路径、候选排名、实际代码、验证、修复链、写回和调用轨迹；本次闭环来自实测 |

## 系统执行链路

`RequirementAgent → Dataset Profile → LLM Knowledge Extraction/bootstrap → Hybrid GraphRAG → PlannerAgent → finite Beam Search → CoderAgent → static gate → Sandbox/ValidatorAgent → CriticAgent/RepairAgent → current-result winner → CuratorAgent → ExplanationAgent/report`。

下一次任务读取 Curator 写入的真实运行和失败经验。AgentRuntime 对各角色的工具分派做权限检查并记录事件；这是有职责、消息契约和证据交换的同步 specialist workflow，不是分布式自主代理群。工作流、图谱、修复和经验学习的 Mermaid 图见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 模型配置与调用记录

正式验收使用本地开放权重，通过 OpenAI-compatible API 实际推理：

| 职责 | 实际模型 |
|---|---|
| 需求理解、知识抽取、规划、Critic、解释 | Qwen2.5-14B-Instruct |
| 完整代码生成、代码修复 | Qwen3-Coder-30B-A3B-Instruct |
| 文档向量检索 | 本地 text2vec-base-chinese 编码器，mean pooling/cosine |

报告逐调用记录 `provider=local_openai_compatible`、model、purpose、latency、token usage、retry count；真实验收没有 Mock 调用。测试中的 Mock fixture 与模型证据明确分开。

四阶段原始报告合计 **58 条真实调用记录**（包含失败/重试和解释恢复，不能全部计为成功调用）。单独的报告知识抽取与事后解释复核还有独立调用记录，未混入这个统计。

早期使用 Transformers 14B 和 NF4 coder；因延迟/超时与上下文限制，后续改用独立 vLLM 0.17.1 服务，最终两个服务均为 32768 context，coder 为 BF16/TP2。模型服务变更不改变早期调用的实际归属。`environment.json`、`backend_transition.json`、`serving_32k_probe.json` 保留过程；未证明早期服务退出根因，不声称已确认 OOM。

### 实测任务

| 阶段 | Run ID | 当前数据实际结果 |
|---|---|---|
| 主 demo，1200 行合成客户数据 | `6bde3f2c39b8` | Logistic Regression PASS；ROC-AUC **0.9288770969**，F1 **0.3934426230**；Gradient Boosting 也 PASS，AUC 0.8928449161 |
| 相似任务，新 1050 行数据与可解释性/延迟偏好 | `9364e416a38a` | Logistic Regression PASS；ROC-AUC **0.8655668976**；Gradient Boosting PASS，AUC 0.8554502370 |
| 专门自修复 | `53ac387cea5c` | Gradient Boosting v1 FAILED → v2 PASS；ROC-AUC **0.8888223211** |
| 跨任务文本分类，16 条记录 | `a054aaacecab` | TF-IDF unigram + Logistic Regression PASS；accuracy **0.50**、weighted F1 **0.3333333333** |

主任务 RF 未通过；没有把失败候选隐藏或标成成功。所有候选尝试见各阶段 JSON。高 ROC-AUC 也不代表默认分类阈值业务效果好：主 winner recall 约 0.261，F1 约 0.393。

文本任务扩展 4 个配置、beam=3；unigram 经 3 轮真实修复后通过，bigram 经 1 轮后通过且指标相同，LLM-proposed 经过 3 轮仍失败。原错误包含 TF-IDF/ColumnTransformer 维度冲突、配置参数错投以及 weighted F1 自报不一致。未降低验证器要求或手改生成源码。该任务明确没有质量阈值，PASS 只表示协议、运行、指标一致性等检查通过；低指标没有证明文本分类质量达标。文本代码契约包含 1D 字符串输入与参数路由规则；本报告中的测量值对应保存的原始代码版本。

## GraphRAG 检索结果

实际查询是“混合数值/类别客户流失预测，ROC-AUC ≥ 0.80，类别不平衡，输出概率”，Task B 另有新客户群、解释性和每行预测延迟约束。

结构化理解产生 binary_classification/churn/字段/指标/约束；Entity Linking 找到 `capability_fe3d4b64611b254d`、`task_binary_classification`、Logistic/GB/RF 等 6 个 anchors。实际返回 **60 个节点、149 条边、8 条 semantic document evidence、6 条历史案例**；backend 明确为 `embedding`。

实际路径示例（括号是遍历方向）：

```text
capability_fe3d4b64611b254d
  ← VALIDATES ← 6bde3f2c39b8                  # 1 hop, relevance 8.892

capability_fe3d4b64611b254d
  → USES_ALGORITHM → algorithm_random_forest
  ← RELATED_TO ← failure_6bde3f2c39b8_..._v4 # 2 hops
```

Serialized Context 包含 `graph_candidates`、`similar_historical_runs`、`failure_and_repair_experiences`、`source_evidence` 与 `system_constraints`，而不是把 GraphML 文件交给 LLM。展示用裁剪案例见 [graph_retrieval_example.json](../examples/acceptance_real_20260907/graph_retrieval_example.json)，完整原始子图在 `second.json`，实际 Planner 输入在 `closed_loop_proof.json`。关系方向、来源和数值属性设计见 [knowledge_graph_schema.md](knowledge_graph_schema.md)。正式抽取结果见 [extracted_knowledge.json](../examples/acceptance_real_20260907/extracted_knowledge.json)，其中 `age → Feature`、`churn → Target`、`ROC-AUC → Metric`；旧错误类型样本已归入历史目录。

## 跨任务经验复用

第一次 retrieval 的 `historical_cases=[]`。运行得到 `6bde3f2c39b8`，Curator 写入实际指标、数据 profile、环境、代码 hash，并保留 RF 的失败及所有替代方案。

第二次运行 `9364e416a38a` 的 retrieval 再次返回 `6bde3f2c39b8`，context similarity 约 **0.98294**。它同时出现在实际 LLM Planner 输入和执行计划的 evidence IDs 中。正式证据为 [closed_loop_proof.json](../examples/acceptance_real_20260907/closed_loop_proof.json)，来源链为真实 Workflow → Curator → 第二次 Retrieval/Planner。`scripts/controlled_prior_injection_demo.py` 仅通过人工注入指标检查 prior 敏感性，不作为真实闭环验收。

| LLM 提议状态的规划分数 | 第一次 | 第二次 |
|---|---:|---:|
| Logistic Regression | 0.7900 | 1.336199 |
| Gradient Boosting | 0.7950 | 0.978267 |
| Random Forest | 0.7940 | 0.649380 |

第二次 Logistic 的评分分解包括 measured historical prior、success/stability、runtime、interpretability 和 exploration bonus（约 0.08485）。两个任务都扩展 **15 个状态，beam=3**，仍实际执行三类算法。历史是 prior，winner 最终由当前验证决定。

这个实验直接证明了写回→再检索→Planner 使用；它不是历史经验的因果消融实验，因为新数据、用户约束和 LLM 输出也发生了变化。另有自动测试在当前非线性数据上实际执行候选，证明即使 fixture 历史偏爱线性模型，当前更合适的非线性算法仍可获胜。

## 代码自修复结果

专门 demo 对真实 LLM 生成的代码明确注入接口错误：将 `predict` 改为 `predict_broken`。此注入只用于证明控制流程，报告有 `DemoFaultInjection` 事件。

```text
v1: missing predict interface → FAILED
  Critic: observed interface mismatch, structured diagnosis
  Repair: real Qwen3-Coder API, diagnosis + strategy + complete code_lines
v2: exact host interface restored → actual sandbox execution → PASS
```

- v1 SHA256：`549c6f01870463641f0182babd499c84e755512ffdb171c5f7990e86f6f77df9`
- v2 SHA256：`1d726e5712c0c51d27d4d7aa37f7143ac13d1043a8133eb42b67796c7699b0ba`
- 实际 repair 调用约 **132.37 秒**，prompt 18719 tokens、output 1233 tokens。
- 原/新代码、原错误、诊断、修复策略、完整两次验证及后续检索在 [self_repair_demo](../examples/acceptance_real_20260907/self_repair_demo/)。

`next_retrieval.json` 证明这次失败经验已被下一次 RetrievalAgent 找到。Graph snapshot 中 RepairExperience 通过 `PRODUCED_VERSION` 连接 v2，v2 保留 parent version。

## 解释校验与报告版本管理

真实 LLM 解释曾出现引用错误、与实测排序矛盾及不受证据支持的历史比较。系统增加动态候选枚举、指标/引用核对与有限比较关系检查。

自修复算法已 PASS、但原解释失败时，仅重试真实解释；原报告完整备份到 `report_revisions`，父报告 SHA 与不可变测量字段由 verifier 核对。其他事后解释复核保存为 `explanation_reviews`，绑定原始报告 semantic hash，Web 明示 review 后的解释，原报告仍保留。未将重新生成的文字伪装成原始运行结果。

历史失败尝试、超时、上下文拒绝、代码换行问题和后端切换日志都保留。`examples/acceptance_20260907/` 是开发过程，不能当作本次四阶段通过报告。

## 自动测试与证据核验

完整 `pytest -q`：**81 passed，0 failed，98 warnings，73.30 秒**。JUnit 与完整控制台结果分别保存在 `test-results.xml`、`pytest.log`；warnings 为当前 sklearn/pandas 组合的弃用提示。

只读 `scripts/verify_evidence.py`：**4 个真实阶段、21 个实际源码版本，全部 hash 和证据断言通过**。检查内容包括真实 provider、源代码 hash、Task A→B Planner 依据、失败经验再次检索、v1 FAILED/v2 PASS、RepairExperience→v2 图关系，以及解释恢复前后的测量字段不变。

从 `git archive HEAD` 导出的独立目录（不含未提交 SQLite/cache）核验通过，见 `clone_verification.json`。原始模型源码证据保留生成时的空格/换行；不会为消除历史 artifact 的 whitespace 提示而改写已核验 hash。

自动测试覆盖实际 sklearn 训练的 A→B 闭环、历史冠军被当前非线性任务淘汰、LLM 无效 JSON 恢复、真实图路径、source grounding、指标造假拒绝、target 泄漏、恶意代码、超时、稳定性失败、资源约束、协议、插件 renderer 与自定义指标。软件测试使用显式 Mock/fixture；真实模型能力的证明是上述独立验收报告。

通过 SSH tunnel 实际检查 Web 的报告选择、第二次任务的复核解释标识、指标、子图筛选、源码以及自修复 v1/v2/hash/parent 展示。范围记录在 `ui_checks.json`，该项属于人工界面验收。

## 主要模块与文件

- `app/llm/`：安全配置、provider、schema negotiation、实际调用计量。
- `app/agents/`：需求、抽取、规划、Critic/Repair、解释与工具分派契约。
- `app/knowledge/`、`app/retrieval/`、`app/experience/`：来源图、真实遍历、serializer、语义融合、案例 prior。
- `app/search/beam.py`、`app/execution/`：状态扩展、候选实际执行、不可变版本链。
- `app/validation/`：可信指标、资源、协议、robustness、安全与隔离。
- `app/api.py`、`app/ui/`：真实报告、子图、代码、修复与解释复核展示。
- `scripts/run_acceptance.py`、`verify_evidence.py`、`acceptance_recovery.py`、`review_explanations.py`：可续跑验收、只读证据核验、原报告保留。
- `tests/`、README、审计/schema/架构文档、示例证据。

## 实现边界

1. 这是 prototype sandbox，AST/audit/resource/unshare 不能证明对任意恶意 Python/原生扩展的生产级隔离；未实现 Docker/VM/cgroups/seccomp 组合。
2. 合成客户数据和极小文本数据不证明真实行业泛化；修复反复观察 holdout 会过拟合。公开文本的独立最终测试见 TEXT_ACCEPTANCE；旧客户流失协议仍需独立最终测试、置信区间与业务成本评估。
3. Beam 是有限一层组合状态扩展与多样性选择；不是 MCTS 或全局最优搜索。经验分数为启发式，重复修复样本相关，未做大规模权重学习。
4. 向量编码使用单条最多 512 tokens；未实现大规模向量索引。Graph linking 仍有规则信号，尚无人工标注的检索评测集。
5. 抽取保留原文跨度并结合 AST，但不等于跨任意真实仓库的全程序语义分析。来源真实不保证 LLM 的所有语义解读正确。
6. 任意自由文本业务约束仍需专用验证插件；解释性目前是规划 prior，不是独立解释质量指标。自然语言解释检查是有限模式，不是通用语义证明。
7. RSS/CPU/runtime 是受控进程观测量，包含依赖载入等开销；不等于单次 estimator 算法复杂度。
8. API 为本机同步单用户原型，无公开鉴权/任务队列/并发图事务服务。多 Agent 是角色协作工作流，未实现分布式自治。

## 异常检测的评价边界

有标签 anomaly detection 使用 F1、Precision、Recall；无标签任务可以执行并输出 `anomaly_score`、`anomaly_rate` 等观测统计。当前 `MetricRegistry` 对无 target 的 anomaly 使用 `runtime_seconds` 进行工程选择；runtime 是运行成本与资源指标，不能在缺少 ground truth 时解释为可靠的算法质量评价。尚未实现完整的 unsupervised quality proxy，属于 Future Work。

## 报告复查

查看封存报告无需启动模型。执行 `python scripts/verify_evidence.py` 可只读核验四阶段证据；执行 `python scripts/verify_text_acceptance.py` 可核验公开文本独立测试。重新生成算法需要按 README 显式配置真实模型服务，并使用新的输出目录。
