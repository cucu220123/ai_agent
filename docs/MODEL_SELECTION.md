# 模型配置与历史基准

## 正式验收配置

| 职责 | 实际模型 |
|---|---|
| 需求、抽取、规划、诊断、解释 | Qwen2.5-14B-Instruct |
| 完整代码生成与修复 | Qwen3-Coder-30B-A3B-Instruct |

正式证据见 [FINAL_ACCEPTANCE](FINAL_ACCEPTANCE.md)、[TEXT_ACCEPTANCE](TEXT_ACCEPTANCE.md) 和 [REAL_LLM_EVIDENCE](REAL_LLM_EVIDENCE.md)。最终 API 服务采用 vLLM，实际 context、dtype 和并行配置记录在验收环境中；早期服务变更保留各自时间范围。

运行方法以 [README](../README.md) 的显式模型配置为准。API 模式使用 OPENAI_MODEL/OPENAI_CODER_MODEL，直接本地推理使用 LOCAL_INSTRUCTION_MODEL_PATH/LOCAL_CODER_MODEL_PATH。适配器中的本地兼容默认值不代表正式 API 验收使用的模型。

## 历史 benchmark

`docs/evidence/benchmark_*.json` 和 `local_model_benchmark.*` 是开发阶段的 1.5B、3B、7B、8B、14B、30B 对照，包含截断、失败和模板 fallback。旧汇总中的模型推荐仅对应当时实验，不作为当前配置或正式验收结论。原始模型输出保留用于审计。

基准通过情况受到提示、上下文预算、推理环境和任务样本影响，不是通用模型排名。正式知识抽取仅引用 [extracted_knowledge.json](../examples/acceptance_real_20260907/extracted_knowledge.json)，旧错误实体类型样本归档在 `docs/evidence/archive/`。
