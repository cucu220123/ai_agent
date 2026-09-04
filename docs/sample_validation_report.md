# 示例验证报告

以下结果来自 `scripts/run_demo.py` 在模拟客户流失数据上的一次运行。报告同时保存 JSON 和 Markdown 两种格式。

| 算法 | 状态 | ROC-AUC | F1 |
|---|---|---:|---:|
| Logistic Regression | passed | 0.8793 | 0.0000 |
| Gradient Boosting | passed | 0.8530 | 0.0000 |
| Random Forest | passed | 0.7927 | 0.0000 |

最终选择 Logistic Regression：满足 ROC-AUC ≥ 0.75，运行耗时较低，并且对运营侧更易解释。验证器同时通过了静态安全、接口导入、独立进程执行、训练/预测功能、输出契约、指标门槛、稳定性和运行时间检查。

说明：模拟数据类别不平衡，固定 0.5 阈值下 F1 较低；生产场景应增加阈值搜索、PR-AUC、代价敏感学习和校准检查。该问题已记录为可扩展的失败/优化经验，而不是将 ROC-AUC 结果误解为分类阈值效果。
