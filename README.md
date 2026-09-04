# AI Algorithm Factory

基于 AI Agent 的算法能力工厂原型，使用“客户流失预测”验证：

```text
自然语言描述 → 能力解析 → 知识检索 → 方案规划 → 代码生成
→ 静态/功能/指标/稳定性验证 → 多轮修复 → 报告 → 知识沉淀
```

项目默认使用离线 Mock/模板模式，因此无网络、无 API 额度时也能完整复现；同时提供 OpenAI-compatible API 和本地 Transformers 适配器。

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

启动 API：

```bash
/data/xiaotianqi/miniconda3/envs/ada_qwen/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000
```

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

```python
train(train_df, target_col, config=None)
predict(model, test_df)
evaluate(model, test_df, target_col)
```

## 知识图谱

节点包括 `Capability`、`Algorithm`、`Dataset`、`FeatureStrategy`、`Metric`、`Environment`、`ValidationRun`、`FailureExperience`、`Constraint`；关系包括 `USES_ALGORITHM`、`VALIDATED_ON`、`REQUIRES_FEATURE`、`EVALUATED_BY`、`REQUIRES`、`VALIDATES`、`RELATED_TO`、`FIXED_BY`。

种子知识在 `app/knowledge/seed_data/knowledge.json`，Schema 说明在 `docs/schema.md`，运行后产生 `knowledge.sqlite` 和 `knowledge.graphml`。

## LLM 配置

默认无需任何配置：

```bash
LLM_PROVIDER=mock
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
export LOCAL_MODEL_PATH=/data/public_checkpoints/huggingface_models/Qwen2.5-1.5B-Instruct
```

较大的本地模型需要空闲 GPU 和显存；项目不会自动下载模型。

## 验证与安全

验证器检查 Python 编译、危险 AST 节点、导入白名单、统一接口、训练预测功能、输出列和概率范围、指标阈值、固定随机种子稳定性及运行时间。生成代码只在项目运行目录中执行；这是原型级防护，生产部署建议使用 Docker/gVisor、只读挂载、禁网和资源配额。

## 已验证结果

在 `data/churn_demo.csv`（1200 行模拟数据）上已实测端到端成功：Logistic Regression ROC-AUC 约 0.879，Gradient Boosting 约 0.853，Random Forest 约 0.793；自动选择 Logistic Regression。每次运行会在 `generated/<run_id>/`、`reports/<run_id>.json` 和 `reports/<run_id>.md` 中保存产物，并回写知识库/GraphML。

## 扩展方向

可继续加入文本分类、异常检测、库存预测插件；向量检索和 Beam Search/MCTS；真实代码仓库抽取；人工审批和版本管理；模型注册、部署配置生成及 Docker 强隔离执行。

