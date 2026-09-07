# Real LLM Evidence

## Cloud API attempt

配置文件：`/data/xiaotianqi/gen_eval/eval/secret.txt`（未复制、未提交、未打印密钥）。

2026-09-07 真实 OpenAI-compatible `models.list()` 调用结果：HTTP 401，服务端明确返回 token quota exhausted。项目记录 endpoint host/provider/status，不记录完整 key；不能把云端调用称为成功。

## Local Qwen2.5-1.5B attempt

模型：`/data/public_checkpoints/huggingface_models/Qwen2.5-1.5B-Instruct`。实际完成一次完整 workflow：

```text
RequirementAgent -> local LLM JSON -> Pydantic validation/correction
RetrievalAgent -> GraphRAG subgraph + semantic evidence + historical cases
AdvisorAgent -> local LLM planning advice
Beam Search -> 3 validated candidate states
CoderAgent -> local LLM code proposal rejected by safety/interface gate
            -> deterministic template fallback
Sandbox/Validator -> PASS
CuratorAgent -> knowledge write-back
```

证据文件：`reports/real_local_llm_evidence.json`、`reports/real_local_llm_repair_evidence.json`；Git 中只保留脱敏摘要 `docs/evidence/real_local_llm_summary.json`，原始 reports 被 `.gitignore` 忽略。

关键事实：RequirementAgent 状态为 `ok`，AdvisorAgent 状态为 `ok`，两者均有 token usage 和 latency。一次运行中三个 CoderAgent metadata 标记为 `template_fallback`；另一次运行中 CoderAgent 标记为 `llm_code_accepted`，但运行时参数契约失败，随后 CriticAgent 和 RepairAgent 真实调用本地 LLM 两轮，仍未通过，最终由显式 template recovery 保证安全完成。这个 fallback/失败过程均被记录，绝不是伪装成 LLM code success。

## Why this is acceptable

真实 LLM 已经进入需求理解、方案建议和证据轨迹；代码生成仍坚持安全门禁。云端额度恢复后，设置 `LLM_PROVIDER=openai` 可以使用同一条路径。更大的本地 Qwen3-Coder 需要约 60GB 权重和较长加载时间，项目保留 adapter 但不默认强制占用 GPU。
