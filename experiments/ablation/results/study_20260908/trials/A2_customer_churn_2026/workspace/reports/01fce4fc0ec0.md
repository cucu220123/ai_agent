# 验证报告 01fce4fc0ec0

任务：CustomerChurnPrediction / binary_classification

状态：**passed**；严格真实模型模式：True

完整证据：[JSON](01fce4fc0ec0.json)

## 当前候选实际比较

| 搜索状态 | 代码来源 | 状态 | 实测指标 | 运行秒数 |
|---|---|---|---|---:|
| algorithm_logistic_regression__llm_proposed | llm | passed | {'accuracy': 0.86, 'f1': 0.5227272727272728, 'precision': 0.71875, 'recall': 0.4107142857142857, 'balanced_accuracy': 0.6869145199063231, 'roc_auc': 0.907347775175644, 'pr_auc': 0.6995153989037561} | 2.825 |
| algorithm_random_forest__llm_proposed | llm | passed | {'accuracy': 0.8233333333333334, 'f1': 0.13114754098360656, 'precision': 0.8, 'recall': 0.07142857142857142, 'balanced_accuracy': 0.5336651053864169, 'roc_auc': 0.881074355971897, 'pr_auc': 0.6211318814493241} | 2.550 |
| algorithm_gradient_boosting__llm_proposed | llm | passed | {'accuracy': 0.8766666666666667, 'f1': 0.6105263157894737, 'precision': 0.7435897435897436, 'recall': 0.5178571428571429, 'balanced_accuracy': 0.7384367681498829, 'roc_auc': 0.897614168618267, 'pr_auc': 0.683939138740142} | 2.764 |

## 为什么选择该方案

```json
{
  "selected_candidate": "algorithm_logistic_regression__llm_proposed",
  "why_this_plan": "The Logistic Regression model was selected due to its simplicity and interpretability, making it a strong baseline model. It achieved a ROC-AUC of 0.9073, meeting the quality threshold of ROC-AUC >= 0.80.",
  "historical_evidence_used": [
    "preprocessing_3b20f945c980b1ea",
    "preprocessing_56a78a1499364981",
    "preprocessing_8817df495c5b2800",
    "preprocessing_8d29697f8be324cb",
    "preprocessing_e30ccf02b0837f02",
    "source_business_material_1c1a6f4d",
    "task_binary_classification"
  ],
  "candidate_comparison": {
    "algorithm_logistic_regression__llm_proposed": "Logistic Regression achieved a ROC-AUC of 0.9073, the highest among the candidates, with a balanced accuracy of 0.6869 and an F1 score of 0.5227.",
    "algorithm_random_forest__llm_proposed": "Random Forest had a lower ROC-AUC of 0.8811 compared to Logistic Regression, with a balanced accuracy of 0.5337 and an F1 score of 0.1311.",
    "algorithm_gradient_boosting__llm_proposed": "Gradient Boosting performed well with a ROC-AUC of 0.8976, a balanced accuracy of 0.7384, and an F1 score of 0.6105."
  },
  "metric_claims": [
    {
      "candidate_id": "algorithm_logistic_regression__llm_proposed",
      "metric": "roc_auc",
      "score": 0.907347775175644
    },
    {
      "candidate_id": "algorithm_random_forest__llm_proposed",
      "metric": "roc_auc",
      "score": 0.881074355971897
    },
    {
      "candidate_id": "algorithm_gradient_boosting__llm_proposed",
      "metric": "roc_auc",
      "score": 0.897614168618267
    }
  ],
  "limitations": [
    "The dataset size is relatively small (900 rows), which may limit the generalizability of the models.",
    "Class imbalance exists in the dataset, with a positive rate of 0.1867, which could affect model performance.",
    "The Random Forest model showed lower performance metrics compared to Logistic Regression and Gradient Boosting."
  ],
  "status": "ok",
  "comparison_check": "passed",
  "provider": "local_openai_compatible",
  "attempts": 2
}
```

## 结构化需求

```json
{
  "raw_description": "根据客户年龄、地区、近30天登录次数、消费金额、投诉次数、会员等级和使用月数预测未来30天客户流失。这是表格二分类，目标字段 churn，必须输出 prediction 和 probability。比较 Logistic Regression、Random Forest、Gradient Boosting 三种候选。评价 ROC-AUC、F1、PR-AUC、Precision、Recall；唯一质量阈值 ROC-AUC >= 0.80。处理缺失值、类别不平衡和未见类别。不要自行增加其他质量阈值。",
  "domain": "customer_behavior_analysis",
  "capability_name": "CustomerChurnPrediction",
  "task_type": "binary_classification",
  "data_type": "tabular",
  "target_column": "churn",
  "feature_columns": [
    "age",
    "region",
    "login_count_30d",
    "total_spend",
    "complaint_count",
    "membership_level",
    "tenure_months"
  ],
  "input_schema": {
    "age": "int64",
    "region": "object",
    "login_count_30d": "float64",
    "total_spend": "float64",
    "complaint_count": "int64",
    "membership_level": "object",
    "tenure_months": "int64"
  },
  "output_schema": {
    "prediction": "int64",
    "probability": "float64"
  },
  "dataset_profile": {
    "dataset_id": "dataset_0d10f1c313632ae5",
    "sha256": "0d10f1c313632ae5f5e405ad063e23b0349eb97532993a599edea55e1ad3ccc8",
    "path": "/data3/xiaotianqi/ai_algorithm_factory/experiments/ablation/results/study_20260908/trials/A2_customer_churn_2026/workspace/data/train.csv",
    "rows": 900,
    "row_count": 900,
    "column_count": 8,
    "columns": [
      "age",
      "region",
      "login_count_30d",
      "total_spend",
      "complaint_count",
      "membership_level",
      "tenure_months",
      "churn"
    ],
    "feature_columns": [
      "age",
      "region",
      "login_count_30d",
      "total_spend",
      "complaint_count",
      "membership_level",
      "tenure_months"
    ],
    "feature_count": 7,
    "dtypes": {
      "age": "int64",
      "region": "object",
      "login_count_30d": "float64",
      "total_spend": "float64",
      "complaint_count": "int64",
      "membership_level": "object",
      "tenure_months": "int64",
      "churn": "int64"
    },
    "missing_rates": {
      "age": 0.0,
      "region": 0.011111111111111112,
      "login_count_30d": 0.012222222222222223,
      "total_spend": 0.025555555555555557,
      "complaint_count": 0.0,
      "membership_level": 0.0,
      "tenure_months": 0.0,
      "churn": 0.0
    },
    "numeric_fraction": 0.7142857142857143,
    "class_count": 2,
    "class_balance": {
      "0": 0.8133333333333334,
      "1": 0.18666666666666668
    },
    "minority_rate": 0.18666666666666668,
    "positive_rate": 0.18666666666666668
  },
  "metrics": [
    "roc_auc",
    "f1",
    "pr_auc",
    "precision",
    "recall"
  ],
  "metric_thresholds": {
    "roc_auc": 0.8
  },
  "output_columns": [
    "prediction",
    "probability"
  ],
  "constraints": [
    "处理缺失值",
    "处理类别不平衡",
    "处理未见类别"
  ],
  "latency_requirement_ms": null,
  "interpretability_requirement": null,
  "resource_constraints": {},
  "probability_output_required": true,
  "class_imbalance": {
    "positive_rate": 0.18666666666666668,
    "is_imbalanced": true
  },
  "candidate_hints": [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting"
  ],
  "uncertainty": [],
  "understanding_confidence": 1.0,
  "candidate_algorithms": [
    "logistic_regression",
    "random_forest",
    "gradient_boosting"
  ],
  "dataset_path": "/data3/xiaotianqi/ai_algorithm_factory/experiments/ablation/results/study_20260908/trials/A2_customer_churn_2026/workspace/data/train.csv"
}
```

## Beam Search：剪枝与探索

```json
{
  "strategy": "combinatorial_beam_search",
  "beam_width": 3,
  "expanded": 15,
  "selected": [
    "algorithm_logistic_regression__llm_proposed",
    "algorithm_random_forest__llm_proposed",
    "algorithm_gradient_boosting__llm_proposed"
  ],
  "scores": {
    "algorithm_logistic_regression__llm_proposed": 0.76,
    "algorithm_logistic_regression__standard_c1": 0.72,
    "algorithm_logistic_regression__balanced_c1": 0.745,
    "algorithm_logistic_regression__regularized_robust": 0.755,
    "algorithm_logistic_regression__weak_regularization": 0.72,
    "algorithm_random_forest__llm_proposed": 0.76,
    "algorithm_random_forest__rf_fast": 0.76,
    "algorithm_random_forest__rf_balanced": 0.745,
    "algorithm_random_forest__rf_deep": 0.745,
    "algorithm_random_forest__rf_regularized": 0.745,
    "algorithm_gradient_boosting__llm_proposed": 0.76,
    "algorithm_gradient_boosting__gb_fast": 0.735,
    "algorithm_gradient_boosting__gb_baseline": 0.72,
    "algorithm_gradient_boosting__gb_slow_deep": 0.72,
    "algorithm_gradient_boosting__gb_shallow": 0.72
  },
  "expansions": [
    {
      "parent_state": "algorithm_logistic_regression",
      "expansion_action": {
        "name": "llm_proposed",
        "parameters": {
          "C": 1.0,
          "max_iter": 100
        },
        "preprocessing_variant": "llm_proposed",
        "preprocessing": [
          "handle_missing_values",
          "encode_categorical_variables"
        ]
      },
      "child_state": "algorithm_logistic_regression__llm_proposed",
      "score": 0.76
    },
    {
      "parent_state": "algorithm_logistic_regression",
      "expansion_action": {
        "name": "standard_c1",
        "parameters": {
          "C": 1.0,
          "class_weight": null,
          "scaler": "standard",
          "threshold": 0.5,
          "numeric_imputer": "median"
        },
        "preprocessing_variant": "imputer=median;scaler=standard",
        "preprocessing": [
          "median imputation",
          "standard"
        ]
      },
      "child_state": "algorithm_logistic_regression__standard_c1",
      "score": 0.72
    },
    {
      "parent_state": "algorithm_logistic_regression",
      "expansion_action": {
        "name": "balanced_c1",
        "parameters": {
          "C": 1.0,
          "class_weight": "balanced",
          "scaler": "standard",
          "threshold": 0.4,
          "numeric_imputer": "median"
        },
        "preprocessing_variant": "imputer=median;scaler=standard",
        "preprocessing": [
          "median imputation",
          "standard"
        ]
      },
      "child_state": "algorithm_logistic_regression__balanced_c1",
      "score": 0.745
    },
    {
      "parent_state": "algorithm_logistic_regression",
      "expansion_action": {
        "name": "regularized_robust",
        "parameters": {
          "C": 0.3,
          "class_weight": "balanced",
          "scaler": "robust",
          "threshold": 0.45,
          "numeric_imputer": "median"
        },
        "preprocessing_variant": "imputer=median;scaler=robust",
        "preprocessing": [
          "median imputation",
          "robust"
        ]
      },
      "child_state": "algorithm_logistic_regression__regularized_robust",
      "score": 0.755
    },
    {
      "parent_state": "algorithm_logistic_regression",
      "expansion_action": {
        "name": "weak_regularization",
        "parameters": {
          "C": 3.0,
          "class_weight": null,
          "scaler": "standard",
          "threshold": 0.5,
          "numeric_imputer": "mean"
        },
        "preprocessing_variant": "imputer=mean;scaler=standard",
        "preprocessing": [
          "mean imputation",
          "standard"
        ]
      },
      "child_state": "algorithm_logistic_regression__weak_regularization",
      "score": 0.72
    },
    {
      "parent_state": "algorithm_random_forest",
      "expansion_action": {
        "name": "llm_proposed",
        "parameters": {
          "n_estimators": 100,
          "max_depth": 5,
          "random_state": 42
        },
        "preprocessing_variant": "llm_proposed",
        "preprocessing": [
          "handle_missing_values",
          "encode_categorical_variables",
          "handle_class_imbalance"
        ]
      },
      "child_state": "algorithm_random_forest__llm_proposed",
      "score": 0.76
    },
    {
      "parent_state": "algorithm_random_forest",
      "expansion_action": {
        "name": "rf_fast",
        "parameters": {
          "n_estimators": 80,
          "max_depth": 6,
          "min_samples_leaf": 2,
          "class_weight": "balanced"
        },
        "preprocessing_variant": "imputer=median;scaler=none",
        "preprocessing": [
          "median imputation"
        ]
      },
      "child_state": "algorithm_random_forest__rf_fast",
      "score": 0.76
    },
    {
      "parent_state": "algorithm_random_forest",
      "expansion_action": {
        "name": "rf_balanced",
        "parameters": {
          "n_estimators": 180,
          "max_depth": 10,
          "min_samples_leaf": 2,
          "class_weight": "balanced"
        },
        "preprocessing_variant": "imputer=median;scaler=none",
        "preprocessing": [
          "median imputation"
        ]
      },
      "child_state": "algorithm_random_forest__rf_balanced",
      "score": 0.745
    },
    {
      "parent_state": "algorithm_random_forest",
      "expansion_action": {
        "name": "rf_deep",
        "parameters": {
          "n_estimators": 240,
          "max_depth": null,
          "min_samples_leaf": 1,
          "class_weight": "balanced_subsample"
        },
        "preprocessing_variant": "imputer=median;scaler=none",
        "preprocessing": [
          "median imputation"
        ]
      },
      "child_state": "algorithm_random_forest__rf_deep",
      "score": 0.745
    },
    {
      "parent_state": "algorithm_random_forest",
      "expansion_action": {
        "name": "rf_regularized",
        "parameters": {
          "n_estimators": 160,
          "max_depth": 7,
          "min_samples_leaf": 5,
          "class_weight": "balanced"
        },
        "preprocessing_variant": "imputer=median;scaler=none",
        "preprocessing": [
          "median imputation"
        ]
      },
      "child_state": "algorithm_random_forest__rf_regularized",
      "score": 0.745
    },
    {
      "parent_state": "algorithm_gradient_boosting",
      "expansion_action": {
        "name": "llm_proposed",
        "parameters": {
          "n_estimators": 100,
          "learning_rate": 0.1,
          "max_depth": 3,
          "random_state": 42
        },
        "preprocessing_variant": "llm_proposed",
        "preprocessing": [
          "handle_missing_values",
          "encode_categorical_variables",
          "handle_class_imbalance"
        ]
      },
      "child_state": "algorithm_gradient_boosting__llm_proposed",
      "score": 0.76
    },
    {
      "parent_state": "algorithm_gradient_boosting",
      "expansion_action": {
        "name": "gb_fast",
        "parameters": {
          "n_estimators": 80,
          "learning_rate": 0.08,
          "max_depth": 2
        },
        "preprocessing_variant": "imputer=median;scaler=none",
        "preprocessing": [
          "median imputation"
        ]
      },
      "child_state": "algorithm_gradient_boosting__gb_fast",
      "score": 0.735
    },
    {
      "parent_state": "algorithm_gradient_boosting",
      "expansion_action": {
        "name": "gb_baseline",
        "parameters": {
          "n_estimators": 120,
          "learning_rate": 0.05,
          "max_depth": 3
        },
        "preprocessing_variant": "imputer=median;scaler=none",
        "preprocessing": [
          "median imputation"
        ]
      },
      "child_state": "algorithm_gradient_boosting__gb_baseline",
      "score": 0.72
    },
    {
      "parent_state": "algorithm_gradient_boosting",
      "expansion_action": {
        "name": "gb_slow_deep",
        "parameters": {
          "n_estimators": 180,
          "learning_rate": 0.03,
          "max_depth": 3
        },
        "preprocessing_variant": "imputer=median;scaler=none",
        "preprocessing": [
          "median imputation"
        ]
      },
      "child_state": "algorithm_gradient_boosting__gb_slow_deep",
      "score": 0.72
    },
    {
      "parent_state": "algorithm_gradient_boosting",
      "expansion_action": {
        "name": "gb_shallow",
        "parameters": {
          "n_estimators": 150,
          "learning_rate": 0.04,
          "max_depth": 2
        },
        "preprocessing_variant": "imputer=median;scaler=none",
        "preprocessing": [
          "median imputation"
        ]
      },
      "child_state": "algorithm_gradient_boosting__gb_shallow",
      "score": 0.72
    }
  ],
  "pruned": [
    {
      "state": "algorithm_random_forest__rf_fast",
      "score": 0.76,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_logistic_regression__regularized_robust",
      "score": 0.755,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_logistic_regression__balanced_c1",
      "score": 0.745,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_random_forest__rf_balanced",
      "score": 0.745,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_random_forest__rf_deep",
      "score": 0.745,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_random_forest__rf_regularized",
      "score": 0.745,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_gradient_boosting__gb_fast",
      "score": 0.735,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_logistic_regression__standard_c1",
      "score": 0.72,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_logistic_regression__weak_regularization",
      "score": 0.72,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_gradient_boosting__gb_baseline",
      "score": 0.72,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_gradient_boosting__gb_slow_deep",
      "score": 0.72,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_gradient_boosting__gb_shallow",
      "score": 0.72,
      "prune_reason": "outside_top_diverse_beam"
    }
  ]
}
```

## 功能、指标、稳定性、资源与鲁棒性

```json
{
  "checks": {
    "static_safety": {
      "passed": true,
      "category": "ok",
      "message": "algorithm protocol passed"
    },
    "interface_import": {
      "passed": true,
      "errors": [],
      "message": "algorithm protocol passed"
    },
    "semantic_contract": {
      "passed": true,
      "errors": [],
      "message": "generated code semantic contract passed"
    },
    "target_leakage": {
      "passed": true,
      "duplicate_target_columns": [],
      "scope": "exact duplicate label detection; semantic leakage needs domain review"
    },
    "evaluation_split": {
      "passed": true,
      "mode": "ablation_development_holdout",
      "training_rows": 900,
      "evaluation_rows": 300,
      "evaluation_profile": {
        "dataset_id": "dataset_8a2348fc630db495",
        "sha256": "8a2348fc630db4955e8449f18a02015241932a3c4fe4eb247eb1fe2a1c2eb86b",
        "path": "/data3/xiaotianqi/ai_algorithm_factory/experiments/ablation/results/study_20260908/trials/A2_customer_churn_2026/development_validation.csv",
        "rows": 300,
        "row_count": 300,
        "column_count": 8,
        "columns": [
          "age",
          "region",
          "login_count_30d",
          "total_spend",
          "complaint_count",
          "membership_level",
          "tenure_months",
          "churn"
        ],
        "feature_columns": [
          "age",
          "region",
          "login_count_30d",
          "total_spend",
          "complaint_count",
          "membership_level",
          "tenure_months"
        ],
        "feature_count": 7,
        "dtypes": {
          "age": "int64",
          "region": "object",
          "login_count_30d": "float64",
          "total_spend": "float64",
          "complaint_count": "int64",
          "membership_level": "object",
          "tenure_months": "int64",
          "churn": "int64"
        },
        "missing_rates": {
          "age": 0.0,
          "region": 0.02666666666666667,
          "login_count_30d": 0.0033333333333333335,
          "total_spend": 0.023333333333333334,
          "complaint_count": 0.0,
          "membership_level": 0.0,
          "tenure_months": 0.0,
          "churn": 0.0
        },
        "numeric_fraction": 0.7142857142857143,
        "class_count": 2,
        "class_balance": {
          "0": 0.8133333333333334,
          "1": 0.18666666666666668
        },
        "minority_rate": 0.18666666666666668,
        "positive_rate": 0.18666666666666668
      },
      "split_seed": 2026,
      "independent_final_test": false
    },
    "isolated_execution": {
      "passed": true,
      "timeout": false,
      "message": "isolated execution passed",
      "environment_sanitized": true,
      "resource_limits_applied": true,
      "runtime_seconds": 2.317710816860199,
      "prediction_rows": 300,
      "resource_limits": {
        "applied": true,
        "cpu_seconds": 90,
        "address_space_mb": 16384,
        "max_file_mb": 16
      },
      "sandbox": {
        "network_audit_blocked": true,
        "filesystem_audit_whitelist": true,
        "production_isolation": false,
        "network_namespace": true
      },
      "robustness": {
        "small_batch": true,
        "missing_values_and_unseen_categories": true,
        "empty_input": "explicitly_rejected",
        "invalid_input": "rejected"
      },
      "max_rss_kb": 1621296,
      "cpu_seconds": 1.44618
    },
    "metric_integrity": {
      "passed": true,
      "source": "parent_process_trusted_metrics",
      "disagreements": {}
    },
    "metrics": {
      "passed": true,
      "per_metric": {
        "roc_auc": true
      },
      "thresholds": {
        "roc_auc": 0.8
      },
      "actual": {
        "accuracy": 0.86,
        "f1": 0.5227272727272728,
        "precision": 0.71875,
        "recall": 0.4107142857142857,
        "balanced_accuracy": 0.6869145199063231,
        "roc_auc": 0.907347775175644,
        "pr_auc": 0.6995153989037561
      }
    },
    "stability": {
      "passed": true,
      "same_seed_drift": 0.0,
      "seeds": [
        42,
        42,
        9
      ],
      "metric_variance": {
        "accuracy": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 3.0814879110195774e-33,
        "balanced_accuracy": 0.0,
        "roc_auc": 1.232595164407831e-32,
        "pr_auc": 0.0
      },
      "normalized_variance": {
        "accuracy": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 3.0814879110195774e-33,
        "balanced_accuracy": 0.0,
        "roc_auc": 1.232595164407831e-32,
        "pr_auc": 0.0
      },
      "variance_limit": 0.02
    },
    "latency": {
      "passed": true,
      "ms_per_row": 0.04347383975982666,
      "limit_ms_per_row": null
    },
    "resource_usage": {
      "passed": true,
      "max_rss_kb": 1621296,
      "cpu_seconds": 1.44618,
      "latency_ms_per_row": 0.04347383975982666,
      "limits": {
        "applied": true,
        "cpu_seconds": 90,
        "address_space_mb": 16384,
        "max_file_mb": 16
      },
      "sandbox": {
        "network_audit_blocked": true,
        "filesystem_audit_whitelist": true,
        "production_isolation": false,
        "network_namespace": true
      }
    },
    "functional": {
      "passed": true,
      "prediction_rows": 300
    },
    "output_contract": {
      "passed": true,
      "required_outputs": [
        "prediction",
        "probability"
      ],
      "metadata": {
        "algorithm": "Logistic Regression",
        "rationale": "Logistic Regression is simple and interpretable, making it a good baseline model.",
        "evidence_ids": [
          "source_business_material_1c1a6f4d",
          "preprocessing_3b20f945c980b1ea",
          "preprocessing_56a78a1499364981",
          "preprocessing_8817df495c5b2800",
          "preprocessing_8d29697f8be324cb",
          "preprocessing_e30ccf02b0837f02",
          "task_binary_classification"
        ]
      }
    },
    "robustness": {
      "passed": true,
      "small_batch": true,
      "missing_values_and_unseen_categories": true,
      "empty_input": "explicitly_rejected",
      "invalid_input": "rejected"
    },
    "runtime_budget": {
      "passed": true,
      "runtime_seconds": 2.8251769691705704,
      "timeout_seconds": 90
    }
  },
  "resource_usage": {
    "max_rss_kb": 1621296,
    "cpu_seconds": 1.44618,
    "latency_ms_per_row": 0.04347383975982666,
    "limits": {
      "applied": true,
      "cpu_seconds": 90,
      "address_space_mb": 16384,
      "max_file_mb": 16
    },
    "sandbox": {
      "network_audit_blocked": true,
      "filesystem_audit_whitelist": true,
      "production_isolation": false,
      "network_namespace": true
    },
    "code_hash": "2e47850f8f67c336128751e5324fbf3b8c7c4a528983a6a023736ded292becae"
  },
  "errors": []
}
```

## 检索到的历史运行

```json
[]
```

## 知识写回与版本

```json
{
  "workflow_run_id": "01fce4fc0ec0",
  "validation_run_ids": [
    "01fce4fc0ec0",
    "01fce4fc0ec0:algorithm_random_forest__llm_proposed:v1",
    "01fce4fc0ec0:algorithm_gradient_boosting__llm_proposed:v1"
  ],
  "algorithm_version_ids": [
    "01fce4fc0ec0_algorithm_logistic_regression__llm_proposed_v1",
    "01fce4fc0ec0_algorithm_random_forest__llm_proposed_v1",
    "01fce4fc0ec0_algorithm_gradient_boosting__llm_proposed_v1"
  ],
  "failure_experience_ids": [],
  "repair_experience_ids": [],
  "capability_id": "capability_186981664f51a7a8"
}
```

## 真实调用计量

```json
[
  {
    "call_id": "llm_0001",
    "purpose": "requirement",
    "timestamp": "2026-09-08T05:24:03.966376+00:00",
    "prompt_sha256": "4de3dc6e4601b9f903c52aff2183254401931668dc8eb3272c20a38760915022",
    "prompt_chars": 3873,
    "status": "ok",
    "response_sha256": "2c04f7f5674e3e022134221891a77fd9a262d29bd220a3bbc572d552036a8c5d",
    "provider": "local_openai_compatible",
    "model": "Qwen2.5-14B-Instruct",
    "latency_ms": 21206.83,
    "token_usage": {
      "prompt_tokens": 1118,
      "completion_tokens": 529,
      "total_tokens": 1647
    },
    "retry_count": 0,
    "generation": {
      "purpose": "requirement",
      "max_new_tokens": 2400,
      "requested_max_new_tokens": 2400,
      "budget_adjustments": [],
      "finish_reason": "stop",
      "truncated": false,
      "eos_reached": true,
      "structured_transport": "json_schema",
      "application_schema_validation": true
    }
  },
  {
    "call_id": "llm_0002",
    "purpose": "planning",
    "timestamp": "2026-09-08T05:25:11.147890+00:00",
    "prompt_sha256": "76f8ca5e75767ba3a2ee851ba3e6f915bed83fb67eeed472325335a7e8433b27",
    "prompt_chars": 9440,
    "status": "ok",
    "response_sha256": "fe04bc2ee308fdae09a6ed0a7f6e95d7f69798a52371dca4876ea00b1e8188a6",
    "provider": "local_openai_compatible",
    "model": "Qwen2.5-14B-Instruct",
    "latency_ms": 21534.73,
    "token_usage": {
      "prompt_tokens": 3086,
      "completion_tokens": 531,
      "total_tokens": 3617
    },
    "retry_count": 0,
    "generation": {
      "purpose": "planning",
      "max_new_tokens": 3200,
      "requested_max_new_tokens": 3200,
      "budget_adjustments": [],
      "finish_reason": "stop",
      "truncated": false,
      "eos_reached": true,
      "structured_transport": "json_schema",
      "application_schema_validation": true
    }
  },
  {
    "call_id": "llm_0003",
    "purpose": "code_generation",
    "timestamp": "2026-09-08T05:25:32.690279+00:00",
    "prompt_sha256": "0595d6b2cc585b6b00e900099eb6631147be678019e7384f56bc947c5232b567",
    "prompt_chars": 15684,
    "status": "ok",
    "response_sha256": "1976a854d27184aa5a63029cd9697ca74f5a79f0e6906aff73295221084fa13e",
    "provider": "local_openai_compatible",
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "latency_ms": 75790.05,
    "token_usage": {
      "prompt_tokens": 4841,
      "completion_tokens": 791,
      "total_tokens": 5632
    },
    "retry_count": 0,
    "generation": {
      "purpose": "code_generation",
      "max_new_tokens": 6400,
      "requested_max_new_tokens": 6400,
      "budget_adjustments": [],
      "finish_reason": "stop",
      "truncated": false,
      "eos_reached": true,
      "structured_transport": "text",
      "application_schema_validation": false
    }
  },
  {
    "call_id": "llm_0004",
    "purpose": "code_generation",
    "timestamp": "2026-09-08T05:26:52.791659+00:00",
    "prompt_sha256": "915406accf54f5bcd3cdbbf83e0a64c958c2de68c0d8687acd501f5b5bbfc369",
    "prompt_chars": 15751,
    "status": "ok",
    "response_sha256": "9aab4aff4b7a86946a2e7fc98e9e904987e66fa9acb4700728b25ad87d3d010e",
    "provider": "local_openai_compatible",
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "latency_ms": 80855.24,
    "token_usage": {
      "prompt_tokens": 4858,
      "completion_tokens": 853,
      "total_tokens": 5711
    },
    "retry_count": 0,
    "generation": {
      "purpose": "code_generation",
      "max_new_tokens": 6400,
      "requested_max_new_tokens": 6400,
      "budget_adjustments": [],
      "finish_reason": "stop",
      "truncated": false,
      "eos_reached": true,
      "structured_transport": "text",
      "application_schema_validation": false
    }
  },
  {
    "call_id": "llm_0005",
    "purpose": "code_generation",
    "timestamp": "2026-09-08T05:28:16.208077+00:00",
    "prompt_sha256": "0192b7c55407de83d24be3bd3ea562c97c525e847dddb631758812bf80334fc4",
    "prompt_chars": 15866,
    "status": "ok",
    "response_sha256": "c557a2fbbe2debded400afc9eaff501ebc3a34547e9208c46f8b21fffcecb65f",
    "provider": "local_openai_compatible",
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "latency_ms": 85597.89,
    "token_usage": {
      "prompt_tokens": 4890,
      "completion_tokens": 883,
      "total_tokens": 5773
    },
    "retry_count": 0,
    "generation": {
      "purpose": "code_generation",
      "max_new_tokens": 6400,
      "requested_max_new_tokens": 6400,
      "budget_adjustments": [],
      "finish_reason": "stop",
      "truncated": false,
      "eos_reached": true,
      "structured_transport": "text",
      "application_schema_validation": false
    }
  },
  {
    "call_id": "llm_0006",
    "purpose": "explanation",
    "timestamp": "2026-09-08T05:30:07.193086+00:00",
    "prompt_sha256": "94b473975275a05a0c7676cb0466e8450f6849131e89468fe824e43fdd34d583",
    "prompt_chars": 16980,
    "status": "ok",
    "response_sha256": "9a83f4239e7c6d81bdbef2c0c169496bca3850439976f7226aeed2e97c21efe7",
    "provider": "local_openai_compatible",
    "model": "Qwen2.5-14B-Instruct",
    "latency_ms": 25112.26,
    "token_usage": {
      "prompt_tokens": 6124,
      "completion_tokens": 577,
      "total_tokens": 6701
    },
    "retry_count": 0,
    "generation": {
      "purpose": "explanation",
      "max_new_tokens": 2600,
      "requested_max_new_tokens": 2600,
      "budget_adjustments": [],
      "finish_reason": "stop",
      "truncated": false,
      "eos_reached": true,
      "structured_transport": "json_schema",
      "application_schema_validation": true
    }
  },
  {
    "call_id": "llm_0007",
    "purpose": "explanation",
    "timestamp": "2026-09-08T05:30:32.307143+00:00",
    "prompt_sha256": "b43a01c2215ad5d1349ed61bf43e760ba19e7f7238a1c45996c8caa123a10065",
    "prompt_chars": 17030,
    "status": "ok",
    "response_sha256": "5327cd1307b9f1a4602d07fa79e134172941fd50459cdac08a4d58880ff77e80",
    "provider": "local_openai_compatible",
    "model": "Qwen2.5-14B-Instruct",
    "latency_ms": 25872.65,
    "token_usage": {
      "prompt_tokens": 6135,
      "completion_tokens": 642,
      "total_tokens": 6777
    },
    "retry_count": 0,
    "generation": {
      "purpose": "explanation",
      "max_new_tokens": 2600,
      "requested_max_new_tokens": 2600,
      "budget_adjustments": [],
      "finish_reason": "stop",
      "truncated": false,
      "eos_reached": true,
      "structured_transport": "json_schema",
      "application_schema_validation": true
    }
  }
]
```

## 修复与不可变代码版本

- 01fce4fc0ec0_algorithm_logistic_regression__llm_proposed_v1: passed, SHA256=2e47850f8f67c336128751e5324fbf3b8c7c4a528983a6a023736ded292becae
- 01fce4fc0ec0_algorithm_random_forest__llm_proposed_v1: passed, SHA256=27105db71a2b3c7a1334d4f59b5c382265235b09edfb3ab9f0ed69a7ad3d69c7
- 01fce4fc0ec0_algorithm_gradient_boosting__llm_proposed_v1: passed, SHA256=fff68674ed5846bd17de3ea715885a446687d6154a0f466577ee7a3c5260bace

## 工作流轨迹

- [RequirementAgent] started
- [RequirementAgent] completed
- [RetrievalAgent] started
- [RetrievalAgent] completed
- [GraphSearch] ok
- [PlannerAgent] started
- [PlannerAgent] completed
- [PlannerAgent] started
- [PlannerAgent] completed
- [PlannerAgent] started
- [PlannerAgent] completed
- [CoderAgent] started
- [CoderAgent] completed
- [CoderAgent] llm_code_accepted
- [Sandbox] started
- [ValidatorAgent] started
- [ValidatorAgent] completed
- [ValidatorAgent] passed
- [CoderAgent] started
- [CoderAgent] completed
- [CoderAgent] llm_code_accepted
- [Sandbox] started
- [ValidatorAgent] started
- [ValidatorAgent] completed
- [ValidatorAgent] passed
- [CoderAgent] started
- [CoderAgent] completed
- [CoderAgent] llm_code_accepted
- [Sandbox] started
- [ValidatorAgent] started
- [ValidatorAgent] completed
- [ValidatorAgent] passed
- [CuratorAgent] started
- [CuratorAgent] completed
- [ExplanationAgent] started
- [ExplanationAgent] completed

限制：当前数据与切分上的实验结果不能保证生产效果；原型沙箱不是对抗恶意代码的生产安全边界。
