# 开发实验与正式证据边界

本目录保存历史 benchmark、结构化输出实验和检索 smoke 记录。原始模型输出用于审计格式及语义失败；旧配置与推荐仅适用于各自实验条件。

正式模型与结果见 [REAL_LLM_EVIDENCE](../REAL_LLM_EVIDENCE.md)。正式知识抽取引用 [extracted_knowledge.json](../../examples/acceptance_real_20260907/extracted_knowledge.json)，正式跨任务闭环仅引用 [closed_loop_proof.json](../../examples/acceptance_real_20260907/closed_loop_proof.json)。

`empty_kg_bootstrap.json` 只记录抽取与检索；`algorithm_retrieval_succeeded` 表示召回算法，不代表训练或 ValidationRunner 通过。该历史文件的字段名已纠正，未重新执行实验。

`controlled_prior_injection.json` 如由同名脚本生成，属于 CONTROLLED TEST ONLY；手工指标只用于 prior 敏感性检查。

历史归档说明见 [archive/README.md](archive/README.md)。
