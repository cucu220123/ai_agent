# 正式闭环与自修复证据

## 跨任务闭环

唯一正式跨任务闭环 proof 为 [closed_loop_proof.json](../examples/acceptance_real_20260907/closed_loop_proof.json)。第一次运行 `6bde3f2c39b8` 实际执行算法并经 Curator 写回；第二次运行 `9364e416a38a` 使用不同数据和约束，再次检索并向 Planner 提供第一次运行。

该文件保留两次规划、搜索和实际 Planner context。完整范围、指标与限制见 [FINAL_ACCEPTANCE](FINAL_ACCEPTANCE.md)。它证明经验使用，但没有进行经验贡献的因果消融。

`scripts/controlled_prior_injection_demo.py` 明确为 **CONTROLLED TEST ONLY**：向临时图中注入人工指标来检查 prior 敏感性，不执行真实 Workflow/Curator/ValidationRunner，不属于正式闭环学习证据。历史输出归档在 `docs/evidence/archive/controlled_prior_injection.json`。

## 自修复

正式记录位于 [self_repair_demo](../examples/acceptance_real_20260907/self_repair_demo/)：真实生成源码被明确注入缺少 `predict` 的接口故障，随后 Critic/Repair 调用真实模型，重新执行后 PASS。before/after 源码、错误、诊断、调用和验证均有记录。

`scripts/closed_loop_learning_demo.py` 和 `scripts/self_repair_demo.py` 是重现实验入口；复现实验必须使用新的输出目录。核查已保存结果使用 `python scripts/verify_evidence.py`，不会重新推理或修改报告。
