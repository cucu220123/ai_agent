# Development-only ablation protocol

此目录只通过实验级适配器组装原有 `AlgorithmFactoryWorkflow`，应用默认行为不变。
七个 YAML 文件使用 JSON 子集语法，可直接用标准库读取，无需额外 YAML 依赖。

## 预注册与隔离

- A0 全部启用；A1 去图；A2 去全部历史经验通道及 seed 数值 prior；A3 原始 Planner Top-K；A4 原始 Planner rank-1 算法；A5 无运行后修复；A6 无检索知识。
- 2 tasks × 3 seeds（42、123、2026），每个设置共享固定 75/25 分层开发划分；顺序用固定 seed 打乱。
- 客户数据是新生成的 1200 条 synthetic population，文本只读取 `data/uci_sentiment/development.csv` 的 2234 条；不执行 745 条 independent final test。
- `fixtures/prior_snapshot.json` 是公开文本最终实验之前的实际知识库只读导出。提取输出与经验固定；不重新抽取，不跨消融任务学习。
- 每次有独立 SQLite、GraphML、generated、reports；Curator 只写该任务私有库。
- `protocol.json` 记录 Git commit、配置、依赖、输入和实现 SHA256；每次任务前后验证正式 evidence、数据和 `app/` 未变。
- 全部触发的 Requirement/Planner/Coder/Critic/Repair/Explanation 使用真实本地 API，温度 0。Extraction 是 frozen real output，不是重新抽取；严格模式禁止 Mock/template 获胜。
- seed 控制数据划分和 API 请求；原 Validator 的稳定性训练 seeds `[42,42,9]` 在所有组相同。GPU/服务可能非确定性，不保证输出字节复现。

## 对照边界

No Beam 不扩展配置，保留 Planner 原始 Top-K；文本只注册一个算法，因此此条件也减少执行预算。Single Candidate 只允许原始 rank-1 算法，仍搜索它的配置。

No Repair 禁止运行失败之后的 Critic/Repair/再验证；所有条件共用 Coder 原有最多两次静态/语义门禁生成预算。首次代码成功必须第一次响应通过门禁且首次 Validator PASS，内部重试另计。实验不注入故障。

LLM-only 保留任务契约、插件白名单、安全验证、搜索与修复预算，Planner 可以引用真实 `current_user_requirement`；此引用不计为检索证据。此条件不是无限制的单次 prompt。

## 运行

先安装根目录依赖，并启动与协议模型名一致的本地 OpenAI-compatible 服务：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:18106/v1
export OPENAI_CODER_BASE_URL=http://127.0.0.1:18108/v1
export EMBEDDING_MODEL_PATH=/path/to/text2vec-base-chinese
python experiments/ablation/run_ablation.py --output experiments/ablation/results/reproduction --prepare-only
# 在运行前提交 protocol 和 inputs，固定实验定义。
python experiments/ablation/run_ablation.py --output experiments/ablation/results/reproduction
python experiments/ablation/summarize_ablation.py --output experiments/ablation/results/reproduction
```

本地模型名为 `Qwen2.5-14B-Instruct` 和 `Qwen3-Coder-30B-A3B-Instruct`。不需要云端凭据。每个 completed result（包括失败）都保留，恢复时跳过；started 但未完成的中断会停止调度，不能自动重跑掩盖失败。

`result.json` 记录指标与分母；`observed.json`、`llm/`、版本源码、执行日志保留完整路径。SQLite 为运行态文件，不提交二进制；初始 JSON、初始/最终 GraphML 与报告可重建知识。质量指标仅成功 winner 统计，n 明示；完成率包含所有已结束任务及其中的失败，行政中断和未启动单列。原计划 3 seeds，只作描述性 mean ± sample std，不声称统计显著。

详见 [实验分析](../../docs/ABLATION_STUDY.md)。正式 acceptance 与独立最终测试结果另行封存，不接受消融反馈。

## 本次研究的提前结束与汇总

`study_20260908` 按预注册随机排程运行至时间预算用尽，共 25 次结束、1 次中断、16 次未启动；客户 14/21、文本 11/21。所有已结束任务均选出了通过验证的算法，但 63 个候选中有 3 个最终失败，不删除任何失败或中断产物。原始 42 次协议保持不变，`study_stop.json` 保存完整状态清单。每组实际 n=1–3，不能宣称完整三种子比较。

```bash
# 只读原始结果，不启动模型，不执行 final test
python experiments/ablation/summarize_ablation.py
```

分析器仅在存在与原协议 SHA256 绑定的停止记录时允许不完整结果；其他意外缺失仍报错。配对分析仅使用共同结束的种子。分析器在停止后增加行政中断处理，执行时原始实现哈希与执行 commit 仍保留；汇总另记录分析脚本 SHA256。

Git clone 不包含两份未跟踪的正式运行 SQLite 时，只读汇总将这两条确切路径列为不可用，不假称验证了不存在的数据库；其他封存源码或数据缺失/改变均拒绝。在原运行服务器上两份数据库与全部核心/正式证据的严格 guard 均通过。

封存研究目录不继续运行；新复现使用新的 output 路径。不可对中断 trial 静默重试，也不可覆盖原始协议或结果。
