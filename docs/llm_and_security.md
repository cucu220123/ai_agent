# LLM 与安全策略

## LLM 调用策略

1. `MockLLM` 是默认路径，保证无网络时可复现；
2. OpenAI-compatible Provider 只提供结构化方案建议和可选代码提案；
3. 本地 Transformers Provider 采用懒加载，不自动下载大模型；
4. LLM 失败、额度耗尽或返回非 JSON/非代码时，记录 trace 并回退模板；
5. LLM 不能直接决定任意导入、任意命令或任意输出接口。

## 代码提案门禁

- 只接受包含 `train`、`predict`、`evaluate` 的 Python 模块；
- 只允许 `numpy/pandas/sklearn/typing/__future__` 导入；
- 禁止 `eval/exec/open/os/subprocess/socket/ctypes` 等危险调用；
- 通过 AST 检查后才会写入候选目录；
- 真正训练和评估在独立 `python -I` 子进程中运行；
- 超时、异常、stdout、stderr、峰值 RSS 都回传到验证报告；
- 生产部署仍应使用容器/虚拟机级隔离、禁网、只读文件系统和资源配额。

