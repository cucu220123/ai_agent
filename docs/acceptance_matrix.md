# 原题要求验收矩阵

## 基础要求

| 原题要求 | 当前实现 | 验证位置 |
|---|---|---|
| 使用开源模型或 LLM 接口 | OpenAI-compatible、本地 Transformers、Mock 三种 Provider | `app/llm/` |
| 具体行业场景 | 客户流失预测，另含文本分类/回归/异常检测 | `data/`、`scripts/` |
| 能力知识库/知识图谱 | SQLite JSON payload + NetworkX + GraphML | `app/knowledge/` |
| 自然语言到可运行代码 | Parser → Planner → Generator → Validator | `app/workflow.py` |
| 统一验证机制 | AST、接口、功能、指标、稳定性、资源、超时 | `app/validation/` |
| CLI/API/UI | CLI、FastAPI、`/ui` | `app/cli.py`、`app/api.py` |
| 完整示例 | `scripts/run_demo.py` 和四类数据集 | `data/`、`scripts/` |
| 文档/依赖/测试 | README、Schema、架构、requirements、pytest | 根目录、`docs/`、`tests/` |

## Agent 闭环

| 流程 | 实现 |
|---|---|
| a 理解输入 | `ParserAgent` + 可选结构化 `AdvisorAgent` |
| b 检索知识 | `RetrieverAgent` + SQLite/NetworkX |
| c 规划方案 | `PlannerAgent` + `BeamSearchPlanner` |
| d 生成代码 | 安全受约束模板 + LLM 代码提案 |
| e 自动测试评估 | `ValidationRunner` + 独立子进程 |
| f 错误修复优化 | `RepairAgent`，最多 N 轮 |
| g 沉淀结果 | `CuratorAgent`，运行记录/候选失败/修复经验入图 |

## 进阶/加分能力

- 多候选算法自动比较：已实现；
- 多轮代码修复：已实现；
- 知识图谱可视化：导出 GraphML；
- 自动验证报告：JSON + Markdown；
- 插件式任务和算法：`PluginRegistry`；
- Beam Search：已实现并记录轨迹；
- 代码安全检查和沙箱执行：AST 白名单 + `python -I` 子进程硬超时；
- 真实文档/代码能力抽取：`python -m app.cli ingest <file-or-dir>`；
- 版本管理：Git 提交、算法源码 SHA-256、元数据 schema 版本；
- 性能/资源分析：运行时间、子进程峰值 RSS；
- 跨场景迁移：表格分类、文本分类、回归、异常检测四种模板；
- 部署配置：Dockerfile、docker-compose、Makefile；
- 多智能体协作：Parser/Retriever/Advisor/Planner/Generator/Validator/Repair/Curator 分工。

