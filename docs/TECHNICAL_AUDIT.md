# AI Algorithm Factory 技术审计

审计日期：2026-09-07  
审计版本：`ebdddd2` 基线；整改结果截至当前 final commits（当前 evidence 见 `docs/REAL_LLM_EVIDENCE.md`、`docs/MODEL_SELECTION.md`）。  
审计方式：阅读全部核心模块、运行测试、运行四类 demo、检查数据库/GraphML/生成产物和 LLM 配置。

## 1. 基线运行结果

- `pytest -q`：当前基线 12 项通过（此前提交基线；本次继续整改前重新运行并确认可复现）。
- 客户流失二分类：可生成 Logistic Regression、Random Forest、Gradient Boosting，并实际训练、比较和报告。
- 文本分类：TF-IDF + Logistic Regression 可以执行，但示例数据少，指标仅具演示意义。
- 回归：Random Forest Regressor 可以执行，MAE/RMSE/R² 已有方向判断。
- 异常检测：Isolation Forest 可以执行，主要报告 anomaly rate/score。
- API：`/health`、`/run`、知识查询、图谱摘要和 `/ui` 可访问。

## 2. 当前真实实现

### 已经进入端到端路径的能力

1. `AlgorithmFactoryWorkflow` 串联 LLM-first 需求解析、Hybrid GraphRAG、规划、组合 Beam、代码生成、验证、Critic/Repair、Curator 和报告。
2. 生成的算法在独立 `python -I` 子进程中运行，父进程接收带标记的 JSON 结果。
3. 静态 AST 检查、危险导入/调用拦截、接口存在性、输出概率范围、行数、指标、稳定性和运行预算检查真实执行。
4. 每个候选算法都会训练和评估，不是只打印方案。
5. ValidationRun、FailureExperience、SourceDocument 等信息确实写入 SQLite，GraphML 也会更新。
6. `auto` Provider 优先云 API、再本地 Qwen、最后安全模板；OpenAI 额度/网络失败会留下 fallback trace。

### 只是规则、模板、Mock 或浅实现的能力

| 能力 | 当前实际情况 | 风险 |
|---|---|---|
| 需求理解 | 正常 workflow 使用 `RequirementUnderstandingAgent` 的 LLM JSON + Pydantic schema + deterministic semantic correction；`ParserAgent` 是 fallback | 小模型仍可能输出低质量字段，需人工/规则校正 |
| 代码生成 | Top beam candidate 真实尝试 LLM code proposal，AST/interface gate 后执行；失败显式触发 Critic/Repair/Template recovery；其余候选按预算模板生成 | 本地 1.5B 代码质量不足，云 API 额度耗尽 |
| 代码修复 | LLM 分支存在，Mock/失败时主要是少量字符串修复和“重新用模板” | 不是完整的诊断-重写-验证闭环 |
| Knowledge Graph | `GraphRetriever` 做实体链接、1~2 hop traversal、path score、SubgraphSerializer；TF-IDF semantic evidence 和 historical cases 合并 | semantic retrieval 是依赖轻量 TF-IDF，不是大型 embedding/FAISS |
| 历史经验使用 | 新 ValidationRun 被写入；下一次只会被浅关键词检索到，没有相似任务统计、成功率、recency、资源和探索分数 | “沉淀”对下一次规划的影响无法证明 |
| Multi-Agent | 多个 class 有职责名，实际工作流主要是一个同步 Python orchestrator；没有统一 Agent I/O schema、工具权限、critic 独立输出 | 职责分离部分成立，但协作协议偏弱 |
| Beam Search | 先有 3 个算法计划，再按分数排序；通常 beam width=3，所以没有有效剪枝的组合搜索 | 搜索创新性和技术含量有限 |
| Hybrid GraphRAG | 没有 embedding/vector retrieval、semantic rerank 或图文证据合并 | 不能称为完整 Hybrid GraphRAG |
| 任务插件 | Registry 已有四类任务声明，但模板/Validator 仍有 task-specific 分支和统一协议缺失 | 新插件接入仍需改核心代码 |
| 安全沙箱 | `python -I`、AST、timeout、RSS 已有；没有真正网络 namespace、cgroup、文件系统白名单或 Docker 执行 | 原型级保护，不能当生产隔离 |
| 解释生成 | 报告展示规则化 rationale 和 LLM note；没有严格要求解释逐条引用检索证据 | 存在泛化或幻觉解释风险 |

## 3. LLM 在主流程中的实际参与

当前 `workflow.py` 会调用 LLM：

1. 一次结构化方案建议调用（要求 JSON，但失败/非 JSON 时忽略）；
2. 该建议可改变候选算法和阈值，但经过白名单过滤；
3. `GeneratorAgent` 在 `provider != mock` 时尝试让 LLM 输出完整 Python 代码，提案不通过静态检查就回退模板；
4. `RepairAgent` 在 `provider != mock` 时尝试让 LLM 根据错误重写完整代码，不合格则走规则修复；
5. 当前真实 demo 使用 Mock 或 API fallback 时，生成代码不是 LLM 生成。

整改后：RequirementAgent/AdvisorAgent 已有真实本地 Qwen 端到端证据；Coder/Repair 也有一次真实 proposal→sandbox failure→Critic→Repair 证据，失败会明确记录并回退。云 API 当前因额度耗尽不能宣称成功。

## 4. Knowledge Graph 是否真正参与 retrieval/planning

整改前部分参与；整改后主路径已升级为 GraphRAG，兼容 fallback 仍保留。

- `KnowledgeStore` 确实维护 NetworkX 图并保存 GraphML。
- `RetrieverAgent` 调用 GraphRetriever、SubgraphSerializer、SemanticRetriever 和 ExperienceRetriever，返回 graph/semantic/history evidence。
- `PlannerAgent` 使用返回的 algorithm `historical_metrics` 和当前 spec 规划。
- GraphRetriever 做实体链接、hop 限制和 path scoring；ValidationRun/Failure/Repair/Version 节点会参与子图证据和历史 prior。

整改后：`GraphRetriever` 先做实体链接，再做 1~2 hop bounded traversal、path score 和 `SubgraphSerializer`；同时 `SemanticRetriever` 做 TF-IDF evidence，`RetrieverAgent` 输出 graph/semantic/history 三类证据。旧 `store.search()` 仍保留作为兼容 fallback。

## 5. 写回后下一次是否真正使用

写回动作是真实的：`CuratorAgent` 插入 `validation_runs`/`experiences` 并建立边。

整改后：`ExperienceRetriever` 计算相似度、success rate、historical score、mean runtime 和 exploration bonus；`PlannerAgent` 把这些 prior 注入 Beam Search，但当前任务仍执行候选。`tests/test_closed_loop.py` 和 `scripts/closed_loop_learning_demo.py` 验证写回后重新检索。

## 6. 代码生成与修复是否真正由 LLM 完成

- 代码生成器可以调用真实 LLM，并接受包含标准接口的代码；但只在非 Mock Provider 且 API 成功、输出可提取、静态门禁通过时发生。
- 模板生成是确定性兜底，也是当前离线 demo 的主要来源。
- 修复器可以把原代码、错误文本交给 LLM，但没有独立 `CriticAgent` 输出 root cause/repair strategy schema；规则修复覆盖面很窄。
- 云 API 仍无成功证据（额度耗尽）；本地 Qwen 已证明真实 Requirement/Advisor 调用，Coder/Repair 的失败和安全 recovery 已记录，不能把模板 recovery 声称为 LLM code success。

## 7. Multi-Agent 职责分离审计

职责命名基本齐全：Parser、Retriever、Advisor、Planner、Generator、Validator、Repair、Curator。问题是：

- Agent 输入输出多数是裸字符串或 dataclass，没有统一协议和版本；
- Validator、Critic、Repair 尚未形成独立的结构化消息链；
- `workflow.py` 仍承担 Provider 调用、候选排序、循环和 winner 选择等较多编排逻辑；
- Agent 没有明确的 tool permission 和 provenance contract；
- 没有可回放的每步 event log。

结论：是“模块化 Agent workflow”，不是完全自治的多智能体系统，需要补齐消息协议、事件轨迹、Critic 和可回放执行。

## 8. Beam Search / Candidate Search 审计

当前 BeamSearchPlanner 只对已生成的少数 AlgorithmPlan 按历史 ROC-AUC、解释性、延迟和资源 penalty 排序。它没有搜索：

- preprocessing 组合；
- class weight/threshold 策略；
- hyperparameter configuration；
- 多层 state expansion；
- 当前任务验证反馈驱动的下一层扩展。

所以当前是“排序器”，不是有明显搜索空间的 Beam Search。

## 9. Sandbox / 安全 / 稳定性限制

已有：AST 禁止危险导入/调用、`python -I`、超时、stdout/stderr 捕获、RSS 记录、固定随机种子重复运行。

仍有限制：

- `python -I` 不是网络隔离；
- 没有 cgroup/rlimit CPU 和地址空间限制；
- 子进程可读取其可访问文件；
- 没有依赖安装隔离和包供应链审计；
- 没有空输入、小批次、未见类别、数据泄漏等完整 robustness matrix；
- 稳定性目前多为同一 seed 的重复运行，不能测跨 seed 方差；
- 任务协议和错误分类还不统一。

## 10. 对照评分标准的 gap

### 技术能力（40%）

- 已有：完整基础闭环、候选执行、自动验证、API、知识写回。
- Gap：真实 LLM 主路径证据、结构化 Requirement Understanding、真正 GraphRAG、历史经验驱动 planning、Critic/Repair 闭环、有效搜索空间。

### 代码质量（30%）

- 已有：模块目录、类型注解、测试、Git、Docker/Makefile。
- Gap：workflow 编排偏重、任务分支硬编码、统一协议和配置 schema 不足、错误分类与事件日志不完整。

### 创新性（15%）

- 已有：多候选、Beam 名义实现、经验写回、代码安全门禁。
- Gap：Beam 未形成组合搜索，GraphRAG/经验学习仍浅，多智能体协作缺少可验证的消息和工具机制。

### 完整性（15%）

- 已有：README、数据、报告、CLI/API/UI、四类任务、测试。
- Gap：真实 LLM 端到端证据、失败修复可重复案例、闭环学习 demo、完整 GraphRAG evidence、架构/验收材料需更新到真实实现。

## 11. 整改优先级

1. 先建立真实 LLM Provider、结构化需求/知识抽取/规划/代码/修复的主路径和 sanitized trace；
2. 实现实体链接、图遍历、子图序列化、语义/图混合检索；
3. 将 ValidationRun/Failure/Repair 形成 ExperienceRetriever 和 prior+exploration scoring；
4. 扩展 Beam 状态为 algorithm + preprocessing + config + threshold；
5. 统一四类任务的 TaskSpec/AlgorithmProtocol/ValidationReport，加入 robustness matrix；
6. 补齐可重复 self-repair、closed-loop learning、真实 API、跨场景和安全测试；
7. 最后更新 README、架构图、验收矩阵和 Git 分阶段提交。
