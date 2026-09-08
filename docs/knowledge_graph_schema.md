# Knowledge Graph schema

## 设计目标

知识图谱表达算法适用条件、相似数据上的验证结果、失败原因与修复策略。SQLite 保存完整 JSON payload 和版本；NetworkX MultiDiGraph 保存有方向、有类型的关系。GraphML 用于可视化与数据交换，Planner 接收经序列化的相关子图。

## 实体

| Entity | 主要字段 | 用途 |
|---|---|---|
| Capability | id, domain, task_type, input/output_schema, capability_version | 面向业务的能力契约 |
| Task | id, task_type | 可比较的任务类别 |
| Algorithm | id, task_types, dependencies, default parameters | 插件与抽取知识链接的算法实体 |
| Dataset | SHA256-based id, rows, feature_count, dtypes, missing rate, class balance | 可测数据上下文 |
| Feature / Target | name, dtype, role | 输入与标签明确区分 |
| InputSchema / OutputSchema | fields and types | 接口描述 |
| PreprocessingStrategy | name, source | 缺失填补、编码、缩放等 |
| Metric | name, direction | 指标定义；数值留在 run |
| Constraint | name, requirement | 指标、资源和接口要求 |
| Dependency / Environment | package/version, Python | 依赖与实际运行环境 |
| HyperparameterConfig | parameters, preprocessing, variant | 候选配置 |
| ValidationRun | algorithm/candidate/version IDs, task/profile, metrics, runtime, resource_usage, status, failure_reason, timestamp, code_hash | 每个候选每轮的真实测量 |
| FailureExperience | failure_type, root_cause, triggering_condition, observed_error, reusable_lesson | 可检索失败案例 |
| RepairExperience | strategy, provider, success, from/to_version, failure_id | 修复策略和实际效果 |
| AlgorithmVersion | code_hash, version, parent_version, source, timestamp | 不可变源码谱系 |
| SourceDocument | path, content hash, chunks, extraction trace | 证据出处 |

FeatureStrategy 保留兼容旧数据；新主流程使用下列规范关系，同时保留 FailureExperience → RELATED_TO → Algorithm 作为直接检索连接。失败的具体运行归属仍由 OCCURRED_IN 表达，RELATED_TO 不替代测量来源。OptimizationExperience 以带 reusable_lesson 的 Failure/RepairExperience 和来源属性表达；适用条件用 Task/Constraint/Preprocessing 及实体属性表达，不为每种文本短语无限新增节点。

## 关系与方向

- Capability SOLVES Task；Capability USES_ALGORITHM Algorithm。
- Algorithm SUITABLE_FOR Task、REQUIRES Dependency、USES_PREPROCESSING PreprocessingStrategy、EVALUATED_BY Metric、VALIDATED_ON Dataset。
- ValidationRun VALIDATES Algorithm、ON_DATASET Dataset、HAS_CONFIG HyperparameterConfig、REQUIRES Environment、SATISFIES Constraint。
- FailureExperience OCCURRED_IN ValidationRun。
- RepairExperience REPAIRS FailureExperience、PRODUCED_VERSION AlgorithmVersion。
- AlgorithmVersion VERSION_OF Algorithm、PARENT_VERSION AlgorithmVersion。
- SourceDocument SUPPORTS 被抽取实体。
- HAS_INPUT / ACCEPTS / OUTPUTS / PREDICTS 表示输入输出角色。

关系名在 schema 枚举中约束；抽取边必须引用同 chunk 实体并有原文跨度。Store 中部分兼容关系允许多种端点，尚非完整本体推理器或 SHACL 引擎。

## 验证指标的属性建模

ROC-AUC=0.86 是某次运行在特定切分、配置和代码版本下的结果，不是算法的固有性质。因此 metrics、runtime_seconds、RSS、CPU、success、timestamp 放在 ValidationRun 属性。Metric 节点定义 roc_auc 的语义，Dataset/Config/Version 关系提供条件，避免 “XGBoost → 0.86” 丢失上下文。

## 来源、标识与可信层级

文件实体使用内容 hash 与规范名称；source 路径、SHA256、chunk、line/character offsets 和 evidence_span 保留。来源跨度经过原文匹配校验。Python AST 提供 imports/functions/classes/signature；LLM 补语义；两者分别保留。

`origin=llm_extracted` 表示文档声称的实验；`origin=measured_workflow` 表示本验证器实际执行。文档中的声称不得变成实测统计；ExperienceRetriever 排除抽取产生的伪测量。seed historical_metrics 只用于冷启动弱 prior。

从文档抽取的 ValidationRun、Failure/RepairExperience、AlgorithmVersion、Config 和 Dataset 使用 source ID、内容 hash、实体类型和局部 ID 构成身份；两个文件都叫“Validation Run”不会覆盖彼此。历史验收快照保留当时的节点身份，不追溯重命名。

同一 workflow 的 winner 最终 ValidationRun 使用 workflow run_id，其余使用 `run_id:candidate:vN`。Planner 仅在已知运行元数据能精确证明等价时，把候选版本别名归一到允许的 canonical run_id，并在 trace 记录；不存在的版本仍拒绝。每轮有独立版本源码和 hash，v2 指向 v1；成功修复后原始失败仍可检索。

## 图检索流程

1. CapabilitySpec 的 task/domain/data_type 和显式算法提示链接 Capability/Task/Algorithm anchors。
2. NetworkX 沿白名单边双向扩展 1–3 hop，记录原方向、遍历方向和完整路径。
3. 相邻节点必须任务兼容；Dependency/Environment/Metric 等 hub 不再无限扩散。
4. 分数由 anchor 匹配、0.72 每 hop 衰减和关系权重组成；截断时保留连接节点。
5. SubgraphSerializer 输出 task、candidate_algorithms、historical_runs、failure_experiences、repair_experiences、nodes、edges。
6. ContextBuilder 区分 requirement、graph_candidates、similar_historical_runs、failure_and_repair_experiences、source_evidence、system_constraints，并按预算删除低排序条目，避免截断无效 JSON。
7. Planner 和 Coder 收到实际序列化 JSON。报告保存精确 Planner 输入和 evidence IDs。

例如实际边方向是 ValidationRun → VALIDATES → Algorithm。检索可以从 Capability → USES_ALGORITHM → Algorithm，逆向 VALIDATES 到历史 run；paths 记录这个方向。超过 hop 预算的失败与修复经验由独立的相似案例通道补充。

## Hybrid 与经验 prior

源文档用本地 Transformer 编码器（mean pooling + cosine） 向量（可选）或 TF-IDF lexical similarity；报告分别标明神经语义检索与词项相似度检索。融合考虑 source similarity、graph distance、task compatibility、recency、validation quality。经验另按领域/目标/特征交集、样本量比例、数值特征占比、类别比例、资源匹配计算 context similarity，再按时间衰减计算 success/stability/runtime/memory prior。

图提供可追溯上下文与 prior，当前任务的实际评估才决定 winner。相关性、衰减与 exploration 系数是可解释启发式，目前未通过大规模离线学习校准。

## 持久化限制

SQLite/NetworkX 适合本地原型；图在写回时导出，payload 有缓存。同一进程中的读写可见；长期运行、多进程并发和超大图需要事务、索引和增量图服务。当前版本提供运行内源代码父子链，尚不提供跨任意仓库的语义版本合并。


