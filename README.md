# AI Algorithm Factory

基于 AI Agent 的算法能力工厂原型，使用“客户流失预测”验证：

```text
自然语言描述 → 能力解析 → 知识检索 → 方案规划 → 代码生成
→ 静态/功能/指标/稳定性验证 → 多轮修复 → 报告 → 知识沉淀
```

项目默认使用 `auto` Provider：先尝试 OpenAI-compatible API，失败后切换本地 Qwen，再由安全模板兜底；`mock` 仍用于 CI/离线测试。这样无网络、无 API 额度时仍可复现，但正常路径会优先尝试真实 LLM。

增强版还支持真实 LLM-first 需求理解、严格 JSON 合约、Hybrid GraphRAG、历史经验 prior + exploration、Critic/Repair 闭环、插件注册表、组合 Beam Search、PR-AUC/最佳 F1 阈值、真实子进程隔离与超时、知识查询 API 和内置极简 Web 页面。模型选择不是默认 1.5B，而是 benchmark 驱动的 14B Instruct + Coder 3B 路由，详见 [MODEL_SELECTION](docs/MODEL_SELECTION.md)。真实 API/本地模型状态和 fallback 原因见 [REAL_LLM_EVIDENCE](docs/REAL_LLM_EVIDENCE.md)。

方案规划使用组合 Beam Search：`algorithm + preprocessing_variant + config_variant` 展开候选，再根据历史 prior、任务约束、资源成本和 exploration bonus 剪枝；当前 validation 仍决定最终 winner。

## 快速开始

建议使用已有的 Python 3.10 环境：

```bash
cd /data/xiaotianqi/ai_algorithm_factory
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/python -m pip install -r requirements.txt
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/python scripts/run_demo.py
```

也可以显式指定需求和数据：

```bash
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/python scripts/generate_demo_data.py
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/python -m app.cli run \
  --description "根据客户年龄、地区、登录频率、消费金额和投诉次数预测客户是否流失，要求 ROC-AUC 不低于 0.75，并输出流失概率" \
  --data data/churn_demo.csv --provider mock
```

运行测试：

```bash
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/python -m pytest -q
```

按原题验收矩阵运行四个示例：

```bash
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/python -m app.cli run --description "根据客户年龄、登录频率和投诉次数预测客户是否流失，要求 ROC-AUC 不低于 0.75" --data data/churn_demo.csv
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/python -m app.cli run --description "对文本评论进行文本分类，预测正面或负面" --data data/text_demo.csv
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/python -m app.cli run --description "根据年龄、访问次数预测消费金额，做回归预测，要求 MAE 不高于 100" --data data/regression_demo.csv
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/python -m app.cli run --description "对设备温度、振动和压力数据进行异常检测" --data data/anomaly_demo.csv
```

启动 API：

```bash
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000
```

可用接口：`/health`、`/run`、`/capabilities`、`/algorithms`、`/plugins`、`/tasks`、`/sources`、`/catalog`、`/knowledge/search?q=客户流失`、`/graph/summary`、`/runs`、`/ui`。API 只允许读取项目目录内的数据文件。

API 示例：

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"description":"预测客户是否流失，要求 ROC-AUC >= 0.75","data_path":"data/churn_demo.csv","provider":"mock"}'
```

## 目录结构

```text
app/agents/       解析、检索、规划、生成、修复、沉淀 Agent
app/generation/   受约束的 sklearn 代码模板
app/knowledge/    SQLite + NetworkX 知识库/知识图谱
app/retrieval/    图遍历、子图序列化、TF-IDF 语义检索
app/experience/   历史案例、闭环学习、自修复演示
app/llm/          Mock、OpenAI-compatible、本地 Transformers 适配器
app/validation/   AST 安全、接口、功能、指标和稳定性验证
app/workflow.py   端到端流程编排
app/cli.py        命令行接口
app/api.py        FastAPI 接口
data/             模拟客户数据和业务材料
docs/             架构和知识图谱 Schema
scripts/          数据生成和一键演示脚本
tests/            单元测试和端到端测试
```

## 场景与算法

示例数据包含年龄、地区、近 30 天登录次数、消费金额、投诉次数、会员等级、使用时长和 `churn` 标签。系统自动比较 Logistic Regression、Random Forest、Gradient Boosting，优先选择满足指标门槛的方案，再按 ROC-AUC 和运行时间排序。生成算法统一提供：

除客户流失表格分类外，项目还提供可运行的文本分类（TF-IDF + Logistic Regression）、回归（Random Forest Regressor）和无监督异常检测（Isolation Forest）模板：

```bash
python -m app.cli run --description "对文本评论进行文本分类，预测正面或负面" --data data/text_demo.csv
python -m app.cli run --description "根据年龄、访问次数预测消费金额，做回归预测" --data data/regression_demo.csv
python -m app.cli run --description "对设备温度、振动和压力数据进行异常检测" --data data/anomaly_demo.csv
```

```python
train(train_df, target_col, config=None)
predict(model, test_df)
evaluate(model, test_df, target_col)
```

## 知识图谱

节点包括 `Capability`、`Algorithm`、`Dataset`、`FeatureStrategy`、`Metric`、`Environment`、`ValidationRun`、`FailureExperience`、`Constraint`；关系包括 `USES_ALGORITHM`、`VALIDATED_ON`、`REQUIRES_FEATURE`、`EVALUATED_BY`、`REQUIRES`、`VALIDATES`、`RELATED_TO`、`FIXED_BY`。

种子知识在 `app/knowledge/seed_data/knowledge.json`，Schema 说明在 `docs/schema.md`，运行后产生 `knowledge.sqlite` 和 `knowledge.graphml`。

原题要求逐项对应关系见 [验收矩阵](docs/acceptance_matrix.md)，LLM 和安全策略见 [安全说明](docs/llm_and_security.md)，架构图见 [ARCHITECTURE](docs/ARCHITECTURE.md)，知识图谱设计见 [knowledge_graph_schema](docs/knowledge_graph_schema.md)，最终验收定义见 [FINAL_ACCEPTANCE](docs/FINAL_ACCEPTANCE.md)。

## LLM 配置

默认 Provider 为 `auto`：先尝试配置的 OpenAI-compatible API，失败后自动切换本地 Qwen；`mock` 只用于 CI/离线测试。

```bash
LLM_PROVIDER=auto
```

OpenAI-compatible 模式：

```bash
export LLM_PROVIDER=openai
export OPENAI_BASE_URL="https://your-endpoint/v1"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="your-model"
```

也支持 `AI_FACTORY_SECRET_FILE=/path/to/secret.txt`，文件可使用 `export OPENAI_BASE_URL=...` 格式。项目不会读取或提交密钥，`.gitignore` 会忽略 `.env`、密钥文件、SQLite 和运行产物。若 API 额度、网络或模型不可用，工作流自动记录原因并降级到确定性模板路径。

本地模型模式：

```bash
export LLM_PROVIDER=local
export LOCAL_INSTRUCTION_MODEL_PATH=/data/public_checkpoints/huggingface_models/Qwen2.5-14B-Instruct
export LOCAL_CODER_MODEL_PATH=/data/public_checkpoints/huggingface_models/Qwen2.5-Coder-3B-Instruct
```

较大的本地模型需要空闲 GPU 和显存；项目不会自动下载模型。

如果要使用你提供的 secret 文件，可设置 `AI_FACTORY_SECRET_FILE=/data/xiaotianqi/gen_eval/eval/secret.txt`；项目也会自动识别该默认路径，但不会把密钥内容写入日志、报告或 Git。

## 验证与安全

验证器检查 Python 编译、危险 AST 节点、导入白名单、统一接口、训练预测功能、输出列和概率范围、指标阈值、固定随机种子稳定性及运行时间。生成代码只在项目运行目录中执行；这是原型级防护，生产部署建议使用 Docker/gVisor、只读挂载、禁网和资源配额。

运行验证会先在父进程做静态检查，再使用 `python -I` 启动独立子进程执行生成代码，并设置硬超时；父进程只接收带标记的 JSON 结果。该隔离仍不是生产级容器沙箱，但已经避免了生成模块直接污染服务进程。

## 已验证结果

在 `data/churn_demo.csv`（1200 行模拟数据）上已实测端到端成功：Logistic Regression ROC-AUC 约 0.879，Gradient Boosting 约 0.853，Random Forest 约 0.793；自动选择 Logistic Regression。每次运行会在 `generated/<run_id>/`、`reports/<run_id>.json` 和 `reports/<run_id>.md` 中保存产物，并把胜者、候选失败和修复经验回写知识库/GraphML。

## 扩展方向

当前已提供目录级 Markdown/Python 能力抽取入口：`python -m app.cli ingest data/`。二分类和异常检测已有可运行模板，回归插件已提供 Random Forest Regressor 模板；后续可继续加入完整文本分类、库存预测算法模板、向量检索和 MCTS。
