# Local Model Selection

本项目不再默认使用 Qwen2.5-1.5B。模型选择依据实际 benchmark，而不是参数量：

- Requirement/Planner/Knowledge Extraction：`Qwen2.5-14B-Instruct`。在严格 JSON/schema gate 下 Requirement/Planner 通过，约 211 秒完成四项基准中的三项；真实 14B extraction 经过三次严格 JSON repair 后产生 13 个实体、4 条 `EVALUATED_BY` 关系。
- Code/Repair：`Qwen2.5-Coder-3B-Instruct`。benchmark 中 Requirement 和 Repair 通过，84.6 秒真实 self-repair demo 通过；初始代码质量仍需要 sandbox/repair gate。
- Lightweight fallback：`Qwen2.5-1.5B-Instruct`。benchmark 四项均失败，不作为默认模型，只保留 CI/无 GPU fallback。
- Optional high-quality exclusive GPU：`Qwen3-Coder-30B-A3B-Instruct`。GPU4 独占 benchmark 约 745 秒、Requirement/Planner/Repair 通过，但共享 GPU0 运行发生 OOM，不能作为常驻默认。
- Qwen3-8B：Requirement/Planner/Repair 通过，但代码输出达到 3072 token 截断，整体延迟约 392 秒，作为 reasoning-heavy optional 模型。

严格 success 定义为 schema、protocol、sandbox runtime 都通过；只产生文本不算成功。完整表格见 `docs/evidence/local_model_benchmark.md`，原始结果 JSON 也保留在同一目录。

## Routing

```text
Requirement / Planner / Extraction -> Qwen2.5-14B-Instruct
Code generation / Repair           -> Qwen2.5-Coder-3B-Instruct
No GPU / CI                         -> Mock or Qwen2.5-1.5B fallback
Exclusive GPU high-quality mode     -> Qwen3-Coder-30B-A3B-Instruct
```

配置：

```bash
export LLM_PROVIDER=local
export LOCAL_INSTRUCTION_MODEL_PATH=/data/public_checkpoints/huggingface_models/Qwen2.5-14B-Instruct
export LOCAL_CODER_MODEL_PATH=/data/public_checkpoints/huggingface_models/Qwen2.5-Coder-3B-Instruct
export LOCAL_LLM_DEVICE=cuda:0
```

