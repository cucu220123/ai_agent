# 补充验收：真实公开文本数据与独立最终测试

原 16 条文本 smoke demo 仅用于流程检查，accuracy 0.50、weighted F1 0.333 不能证明分类质量。本页记录公开语料实验 `683902e2a7cf`；原始证据保留，不替换或重新标注旧结果。

## 数据与预先固定的实验协议

使用 [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)，引用 Kotzias (2015)，DOI [10.24432/C57604](https://doi.org/10.24432/C57604)，许可 CC BY 4.0。来源包括产品、电影和餐馆评论。原始 3000 条、来源文件和行号、许可与下载 archive SHA256 均保留在 `data/uci_sentiment/`。

解析修复了两个真实问题：CSV 引号规则会吞并 IMDB 记录；`str.splitlines()` 会把句内特殊换行字符误当作新记录。现在按文件 LF 分记录、最后一个 TAB 分标签。对规范化文本（NFKC/casefold/空白）去重后得到 **2979 条**，没有发现标签冲突组。

固定 seed=20260908，按来源和类别分层：

- **开发集 2234 条**：实际 Agent 只收到这份数据；候选内部训练 1675 条、验证 559 条。
- **最终测试集 745 条**：保持在 agent workspace 之外；不进入 Requirement、Retrieval、Planner、Coder、Critic 或 Repair 输入。
- 两个集合没有规范化特征行重叠，标签不参与重叠判定。
- 预先规定开发/最终 accuracy 和 weighted F1 均至少 **0.70**；beam=3、最大修复轮数=3。配置和数据 hash 在运行前写入 `evaluation_policy.json`，续跑不能改变。

## 实际模型与经验复用

Qwen2.5-14B-Instruct 完成理解、规划和解释；Qwen3-Coder-30B-A3B-Instruct 生成三个完整算法模块。通过本机 vLLM/OpenAI-compatible API 实际推理，**7 次调用记录、全部 API 返回成功**；需求 JSON 有一次应用侧恢复。没有 Mock/模板候选。

实验复制既有真实验收知识库作为起点，保留初始 GraphML hash。三个材料的抽取结果命中内容 hash 缓存，来源是此前真实 LLM 抽取，该阶段没有新增 Extraction API 调用。正式源抽取证据为 [extracted_knowledge.json](../examples/acceptance_real_20260907/extracted_knowledge.json)。

GraphRAG 返回 **64 个节点、203 条边、8 条神经向量证据、10 条历史案例和 8 条失败经验**。旧文本运行 `a054aaacecab` 及其各版本被重新找到，context similarity 约 0.744421；包括 TF-IDF 参数错投、预处理维度、weighted F1 不一致的失败。实际 Planner context 和生成代码 metadata 引用了这些经验。

这证明旧经验确实进入新任务；不能据此把当前效果归因于某一条经验，因为数据规模、代码契约和模型输出也变化了，尚未做因果消融。

## 候选与测量

同一个文本算法插件扩展 4 个配置，beam 选择 3 个并全部实际执行。三个候选均 v1 通过，该运行无需修复。

| 当前开发集候选 | Accuracy | Weighted F1 | 状态 |
|---|---:|---:|---|
| LLM proposed | 0.799642 | 0.799638 | PASS |
| TF-IDF unigram | 0.794275 | 0.794274 | PASS |
| TF-IDF bigram | **0.801431** | **0.801424** | PASS / selected |

选定 bigram 代码后，`FinalHoldoutEvaluator` 先写入独占 commitment，绑定原始选择报告、源码、开发/最终数据 hash；冻结源码为 `frozen_winner.py`。然后用全部 2234 条开发数据重新训练，在 745 条最终数据上执行同一套可信验证。

**最终结果：accuracy 0.8241610738，weighted F1 0.8241382607，balanced accuracy 0.8241769725，PASS。** 原代码 evaluate 与父进程重算指标一致，功能、稳定性、资源和 robustness 检查通过。最终分数没有回到 Agent，也没有根据最终分数更换 winner 或修复代码。

重复调用相同 commitment 只返回已有结果；更换代码/数据/选择报告会拒绝；中断后保留 ledger 并拒绝自动重评。本次为补充 Web companion 再次调用 final 阶段时返回的就是缓存，未重复执行最终测量。

开发阶段 LLM 解释在最终测试执行前生成，其“final test unavailable”描述只对应 Agent 当时没有最终数据/结果。页面将其与后续独立最终测试结果分开显示；不能把开发解释当作最终测试说明。

## 验证与重现

全量测试 **86 passed，98 warnings，67.84 秒**；新覆盖包括真实子进程全量开发数据训练、外部最终数据行数/指标、重复文本泄漏、代码篡改、重复评估/中断保护、API hash 绑定、3000 条来源记录与切分完整性。

`scripts/verify_text_acceptance.py` 只读核验通过：**3 个不可变代码版本、7 条真实调用、冻结 winner、数据/报告 hash、时间顺序和两个阶段的阈值**。

另从已推送提交 `101fc97` 用 `git archive` 导出全新临时目录，独立运行同一核验也通过，不依赖未提交 SQLite 或临时产物；记录在 `clone_verification.json`。生成源码保留模型输出的原始字节（包括空白），以维持已验证的代码 hash。

```bash
# 已提交原数据、固定切分和来源；可选重新下载并检查固定 archive hash
python scripts/prepare_sentiment_data.py

# 配置真实 API 后，用新输出目录运行；不需要旧知识库也可冷启动
python scripts/run_text_acceptance.py --provider openai --output examples/my_text_trial
python scripts/verify_text_acceptance.py --output examples/my_text_trial

# 复查本次已提交证据，无需加载模型
python scripts/verify_text_acceptance.py --output examples/acceptance_text_20260908

# 查看本次报告
AI_FACTORY_WORKSPACE="$PWD/examples/acceptance_text_20260908/workspace" \
  uvicorn app.api:app --host 127.0.0.1 --port 18081
```

本次命令另指定 `--prior-workspace examples/acceptance_real_20260907/workspace`，通过 SQLite backup 复用服务器上的旧真实运行。SQLite 不提交 Git，因此 clone 后可以冷启动；初始 GraphML 与历史引用保留供审核。API 只有在 final companion 的选择报告 hash 匹配时才附加最终结果，开发集指标保持原值。


## 边界

该实验验证的是有标签文本分类。其他任务的评价边界：有标签 anomaly detection 使用 F1、Precision、Recall；无标签任务可以执行并输出 `anomaly_score`、`anomaly_rate` 等观测统计。当前 `MetricRegistry` 对无 target 的 anomaly 使用 `runtime_seconds` 进行工程选择；runtime 是运行成本与资源指标，不能在缺少 ground truth 时解释为可靠的算法质量评价。尚未实现完整的 unsupervised quality proxy，属于 Future Work。

两项实验的数据和协议不同，**不能把旧 smoke demo 的 0.333 与公开语料的 0.824 当作同一数据上的改进幅度**。最终拟合使用更多开发数据，开发与最终分数也不是同一切分的直接比较。候选差异很小，未进行显著性检验，不能声称 bigram 普遍最好。

最终 ledger 是可复现性保护，不能阻止操作者删除目录或另起实验来人为反复看分数；不是对恶意操作者的访问控制。原型沙箱限制仍适用。最终阶段保留重复训练以检查稳定性，但没有候选搜索或修复反馈。公开小语料的混合来源分层切分不等于生产测试或跨来源泛化评测；未做置信区间、外部新数据或 adversarial leakage 评测。

证据目录：[examples/acceptance_text_20260908](../examples/acceptance_text_20260908/)。
