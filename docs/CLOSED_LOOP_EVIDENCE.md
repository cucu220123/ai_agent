# Closed-loop and Self-repair Evidence

## Self-repair

运行：

```bash
python scripts/self_repair_demo.py
```

该 demo 故意把 `predict` 改成 `predict_broken`：

```text
first validation: FAILED (missing function: predict)
CriticAgent: interface_failure / restore strict algorithm protocol
RepairAgent: restore required predict interface
second validation: PASSED
```

详细 before/broken/after code、诊断、修复事件和第二次 ValidationResult 保存在 `examples/self_repair_demo/result.json`。

## Experience learning

运行：

```bash
python scripts/closed_loop_learning_demo.py
```

流程是：读取运行前历史 → 执行任务 → Curator 写入 ValidationRun/AlgorithmVersion/HyperparameterConfig → 再次 GraphRAG/ExperienceRetriever → 输出匹配的 run id、子图和历史 prior。测试 `tests/test_closed_loop.py` 对写回后重新检索做了断言。

