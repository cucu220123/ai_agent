# 示例验证报告

以下结果来自 `scripts/run_demo.py` 在 1200 行模拟客户流失数据上的一次运行。报告同时保存 JSON 和 Markdown 两种格式。

| 算法 | 状态 | ROC-AUC | F1 |
|---|---|---:|---:|
| Logistic Regression | passed | 0.9259 | 0.3934 |
| Gradient Boosting | passed | 0.9080 | 0.3846 |
| Random Forest | passed | 0.8567 | 0.3636 |

最终选择 Logistic Regression：满足 ROC-AUC ≥ 0.75，运行耗时较低，并且对运营侧更易解释。验证器同时通过了静态安全、接口导入、独立进程执行、训练/预测功能、输出契约、指标门槛、稳定性和运行时间检查。

说明：模拟数据存在类别不平衡，报告同时给出 PR-AUC、最佳 F1 和最佳阈值。生产场景应结合业务成本做阈值选择、代价敏感学习和概率校准，而不是只看 ROC-AUC。
