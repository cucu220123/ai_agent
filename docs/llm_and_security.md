# LLM 与安全策略

## 模型调用

默认 auto 使用显式配置的真实 API 或可用本地模型；没有真实 provider 时明确失败，不静默转为 Mock。正常严格路径要求需求理解、知识抽取、规划及获胜源码来自真实模型。Mock 和模板仅适用于显式离线测试或显式允许的 fallback。

正式模型、原始调用与实测范围见 [REAL_LLM_EVIDENCE](REAL_LLM_EVIDENCE.md)。结构化输出经过 schema 与语义检查；错误保留并进行有界重试。

## 执行边界

- 生成模块遵守 train/predict/evaluate/metadata 协议，需要概率时还需 predict_proba。
- AST、导入和接口检查限制生成代码；同一检查也应用于修复结果。
- 算法在临时目录中的隔离子进程运行；检查超时、资源与输出。
- 可信父进程重算指标，拒绝生成代码自报的不一致分数。
- 清理子进程中的凭证环境；使用 audit 和可用的 Linux 网络命名空间增强限制。

这是 prototype sandbox。AST、audit 和 resource limits 不能证明任意 Python/原生扩展安全；生产隔离需要低权限容器/VM、只读挂载、cgroups、seccomp 与禁网。

## 配置安全

API 配置由环境变量或显式指定的外部配置文件提供，不使用个人机器上的默认凭证路径。实际凭证不进入源码、文档、报告或 Git；日志中的端点使用指纹，保留模型、耗时、usage 和错误类型。`.gitignore` 排除环境文件和凭证文件。
