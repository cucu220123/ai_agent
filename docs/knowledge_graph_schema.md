# Knowledge Graph Schema

## Design goals

图谱不是 GraphML 展示文件，而是用来支持实体链接、1~2 hop 路径检索、历史验证证据和失败/修复经验复用。数值型指标不单独建节点，保存在 `ValidationRun` 属性中，避免图膨胀并保留一次运行的上下文。

## Entities

| Entity | Main properties | Meaning |
|---|---|---|
| Capability | id, domain, task_type, input/output schema | 可复用算法能力 |
| Task | id, task_type, data_type | 任务语义 |
| Algorithm | id, name, resource profile | 算法族 |
| AlgorithmVersion | version, code_hash, timestamp, status | 代码版本 |
| Dataset | path, target, row_count, profile | 数据和画像 |
| Feature | name, dtype, missing_rate | 输入特征 |
| PreprocessingStrategy | id, steps | 预处理策略 |
| Metric | name, direction, threshold | 评价指标定义 |
| Constraint | name, value | 业务/资源限制 |
| Dependency | name, version | 软件依赖 |
| Environment | python, dependencies, hardware | 运行环境 |
| HyperparameterConfig | parameters, variant | 运行配置 |
| ValidationRun | metrics, runtime, memory, status, timestamp | 一次验证事实 |
| FailureExperience | failure_type, root_cause, triggering_condition | 失败经验 |
| RepairExperience | strategy, diagnosis, success | 修复经验 |
| SourceDocument | source, kind, provenance | 原始材料 |

## Relations

`Capability-SOLVES->Task`、`Capability-USES_ALGORITHM->Algorithm`、`Algorithm-SUITABLE_FOR->Task`、`Algorithm-REQUIRES->Dependency`、`Algorithm-USES_PREPROCESSING->PreprocessingStrategy`、`Algorithm-EVALUATED_BY->Metric`、`Capability-VALIDATED_ON->Dataset`、`ValidationRun-VALIDATES->Algorithm`、`ValidationRun-ON_DATASET->Dataset`、`ValidationRun-HAS_CONFIG->HyperparameterConfig`、`FailureExperience-OCCURRED_IN->ValidationRun`、`RepairExperience-REPAIRS->FailureExperience`、`SourceDocument-SUPPORTS->Capability/Algorithm/Experience`、`AlgorithmVersion-VERSION_OF->Algorithm`。

## Provenance

所有 LLM/AST 抽取结果保留 `source` 和 `provenance.evidence_spans`；运行结果保留 run id、代码 hash、时间、环境、数据集画像和修复历史。Planner 使用这些事实计算 prior，但当前任务仍会执行候选方案，保留 exploration bonus，避免历史最佳算法永久垄断。

