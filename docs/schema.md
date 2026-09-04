# 能力知识图谱 Schema

## 节点

| 节点类型 | 关键字段 | 用途 |
|---|---|---|
| Capability | id, name, task_type, input_schema, output_schema | 描述可复用算法能力 |
| Algorithm | id, name, task_types, preprocessing, historical_metrics | 算法模板和经验指标 |
| Dataset | id, path, columns, target, row_count | 训练/验证数据 |
| FeatureStrategy | id, name, steps | 缺失值、编码、缩放等策略 |
| Metric | id, name, direction, threshold | 统一指标和门槛 |
| Environment | id, python, dependencies | 可复现依赖环境 |
| ValidationRun | id, status, metrics, checks, runtime | 每次自动验证记录 |
| FailureExperience | id, kind, summary, repair_history | 失败案例和可复用修复经验 |
| Constraint | id, name, value | 业务、资源和接口约束 |

## 关系

`Capability -USES_ALGORITHM-> Algorithm`、`Capability -VALIDATED_ON-> Dataset`、`Capability -REQUIRES_FEATURE-> FeatureStrategy`、`Algorithm -EVALUATED_BY-> Metric`、`Algorithm -REQUIRES-> Environment`、`ValidationRun -VALIDATES-> Capability`、`FailureExperience -RELATED_TO-> Algorithm`、`FailureExperience -FIXED_BY-> RepairStrategy`。

## 存储

- SQLite：保留完整 JSON payload，支持审计和查询；
- NetworkX：构建有向属性多重图；
- GraphML：导出 `knowledge.graphml`，可用 Gephi/Cytoscape 查看；
- JSON：`app/knowledge/seed_data/knowledge.json` 作为可版本管理的种子知识。

