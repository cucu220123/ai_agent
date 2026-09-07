# 验证报告 2b2a3d8a9bac

任务：Customer Churn Prediction / binary_classification

状态：**failed**；严格真实模型模式：True

完整证据：[JSON](2b2a3d8a9bac.json)

## 当前候选实际比较

| 搜索状态 | 代码来源 | 状态 | 实测指标 | 运行秒数 |
|---|---|---|---|---:|
| algorithm_logistic_regression__llm_proposed | llm | failed | {} | 0.284 |

## 为什么选择该方案

```json
{
  "status": "deterministic_fallback",
  "selected_candidate": "algorithm_logistic_regression__llm_proposed",
  "why_this_plan": "Candidates were filtered by task/constraints, prioritized with contextual history and exploration, and ranked using current independently recomputed validation metrics.",
  "historical_evidence_used": [
    "6bde3f2c39b8",
    "6bde3f2c39b8:algorithm_gradient_boosting__llm_proposed:v1",
    "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v1",
    "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v2",
    "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v3",
    "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v4",
    "9364e416a38a",
    "9364e416a38a:algorithm_gradient_boosting__llm_proposed:v1",
    "9364e416a38a:algorithm_random_forest__llm_proposed:v1"
  ],
  "candidate_comparison": {
    "algorithm_logistic_regression__llm_proposed": "failed"
  },
  "metric_claims": [],
  "limitations": [
    "Synthetic demo data; holdout metrics are not production performance.",
    "Prototype process/audit sandbox is not a production hostile-code security boundary.",
    "Generated free-text explanations require human review; numeric claims and evidence ids are checked."
  ],
  "error": "RuntimeError: APIConnectionError: Connection error."
}
```

## 结构化需求

```json
{
  "raw_description": "Build customer churn prediction for mixed numeric/categorical tabular customer data. Target column churn, binary 0/1. Exclude churn from features. Require ROC-AUC >= 0.80 and positive-class probability. Report F1, precision, recall; handle missing values, unseen categories and class imbalance. Compare logistic regression, random forest and gradient boosting.",
  "domain": "customer analytics",
  "capability_name": "Customer Churn Prediction",
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
    "prediction": "float",
    "probability": "float"
  },
  "dataset_profile": {
    "dataset_id": "dataset_4bf999a657407039",
    "sha256": "4bf999a657407039780c32ce01e9ca9cd2d1fdafa131747b9ebb4e8ef8ef8e45",
    "path": "/data3/xiaotianqi/ai_algorithm_factory/examples/acceptance_real_20260907/workspace/data/churn_first.csv",
    "rows": 1200,
    "row_count": 1200,
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
      "region": 0.015,
      "login_count_30d": 0.01,
      "total_spend": 0.025,
      "complaint_count": 0.0,
      "membership_level": 0.0,
      "tenure_months": 0.0,
      "churn": 0.0
    },
    "numeric_fraction": 0.7142857142857143,
    "class_count": 2,
    "class_balance": {
      "0": 0.8475,
      "1": 0.1525
    },
    "minority_rate": 0.1525,
    "positive_rate": 0.1525
  },
  "metrics": [
    "roc_auc",
    "f1",
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
    "Exclude churn from features",
    "Handle missing values",
    "Handle unseen categories",
    "Handle class imbalance"
  ],
  "latency_requirement_ms": null,
  "interpretability_requirement": null,
  "resource_constraints": {},
  "probability_output_required": true,
  "class_imbalance": {
    "positive_rate": 0.1525,
    "is_imbalanced": true
  },
  "candidate_hints": [
    "logistic_regression",
    "random_forest",
    "gradient_boosting"
  ],
  "uncertainty": [
    "The exact distribution of class imbalance is unknown.",
    "The dataset may contain unseen categories in the test set."
  ],
  "understanding_confidence": 1.0,
  "candidate_algorithms": [
    "logistic_regression",
    "random_forest",
    "gradient_boosting"
  ],
  "dataset_path": "/data3/xiaotianqi/ai_algorithm_factory/examples/acceptance_real_20260907/workspace/data/churn_first.csv"
}
```

## Beam Search：剪枝与探索

```json
{
  "strategy": "combinatorial_beam_search",
  "beam_width": 1,
  "expanded": 14,
  "selected": [
    "algorithm_logistic_regression__llm_proposed"
  ],
  "scores": {
    "algorithm_logistic_regression__llm_proposed": 1.057644,
    "algorithm_logistic_regression__standard_c1": 0.992644,
    "algorithm_logistic_regression__balanced_c1": 1.017644,
    "algorithm_logistic_regression__regularized_robust": 1.027644,
    "algorithm_logistic_regression__weak_regularization": 0.992644,
    "algorithm_gradient_boosting__llm_proposed": 1.016199,
    "algorithm_gradient_boosting__gb_fast": 0.991199,
    "algorithm_gradient_boosting__gb_baseline": 0.976199,
    "algorithm_gradient_boosting__gb_slow_deep": 0.976199,
    "algorithm_gradient_boosting__gb_shallow": 0.976199,
    "algorithm_random_forest__rf_fast": 0.61445,
    "algorithm_random_forest__rf_balanced": 0.59945,
    "algorithm_random_forest__rf_deep": 0.59945,
    "algorithm_random_forest__rf_regularized": 0.59945
  },
  "expansions": [
    {
      "parent_state": "algorithm_logistic_regression",
      "expansion_action": {
        "name": "llm_proposed",
        "parameters": {
          "C": 1.0,
          "max_iter": 100,
          "class_weight": "balanced"
        },
        "preprocessing_variant": "llm_proposed",
        "preprocessing": [
          "handle_missing_values",
          "encode_categorical_variables",
          "handle_class_imbalance"
        ]
      },
      "child_state": "algorithm_logistic_regression__llm_proposed",
      "score": 1.057644
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
      "score": 0.992644
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
      "score": 1.017644
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
      "score": 1.027644
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
      "score": 0.992644
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
      "score": 1.016199
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
      "score": 0.991199
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
      "score": 0.976199
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
      "score": 0.976199
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
      "score": 0.976199
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
      "score": 0.61445
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
      "score": 0.59945
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
      "score": 0.59945
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
      "score": 0.59945
    }
  ],
  "pruned": [
    {
      "state": "algorithm_logistic_regression__regularized_robust",
      "score": 1.027644,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_logistic_regression__balanced_c1",
      "score": 1.017644,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_gradient_boosting__llm_proposed",
      "score": 1.016199,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_logistic_regression__standard_c1",
      "score": 0.992644,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_logistic_regression__weak_regularization",
      "score": 0.992644,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_gradient_boosting__gb_fast",
      "score": 0.991199,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_gradient_boosting__gb_baseline",
      "score": 0.976199,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_gradient_boosting__gb_slow_deep",
      "score": 0.976199,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_gradient_boosting__gb_shallow",
      "score": 0.976199,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_random_forest__rf_fast",
      "score": 0.61445,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_random_forest__rf_balanced",
      "score": 0.59945,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_random_forest__rf_deep",
      "score": 0.59945,
      "prune_reason": "outside_top_diverse_beam"
    },
    {
      "state": "algorithm_random_forest__rf_regularized",
      "score": 0.59945,
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
      "passed": false,
      "category": "interface",
      "message": "missing function: predict"
    },
    "functional": {
      "passed": false,
      "message": "missing function: predict"
    },
    "runtime_budget": {
      "passed": true,
      "runtime_seconds": 0.28384560346603394,
      "timeout_seconds": 90
    }
  },
  "resource_usage": {
    "code_hash": "ef7b1b8b1ed5c6811b1689b4b260990980e21e5316d16a92cd8b6caf4d67942d"
  },
  "errors": [
    "ValueError: missing function: predict"
  ]
}
```

## 检索到的历史运行

```json
[
  {
    "run_id": "6bde3f2c39b8",
    "algorithm_id": "algorithm_logistic_regression",
    "similarity": 1,
    "metrics": {
      "accuracy": 0.8766666666666667,
      "f1": 0.39344262295081966,
      "precision": 0.8,
      "recall": 0.2608695652173913,
      "balanced_accuracy": 0.6245292707976721,
      "roc_auc": 0.9288770968846285,
      "pr_auc": 0.7114689013236358
    }
  },
  {
    "run_id": "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v4",
    "algorithm_id": "algorithm_random_forest",
    "similarity": 1,
    "metrics": {}
  },
  {
    "run_id": "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v3",
    "algorithm_id": "algorithm_random_forest",
    "similarity": 1,
    "metrics": {}
  },
  {
    "run_id": "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v2",
    "algorithm_id": "algorithm_random_forest",
    "similarity": 1,
    "metrics": {}
  },
  {
    "run_id": "6bde3f2c39b8:algorithm_random_forest__llm_proposed:v1",
    "algorithm_id": "algorithm_random_forest",
    "similarity": 1,
    "metrics": {}
  },
  {
    "run_id": "6bde3f2c39b8:algorithm_gradient_boosting__llm_proposed:v1",
    "algorithm_id": "algorithm_gradient_boosting",
    "similarity": 1,
    "metrics": {
      "accuracy": 0.87,
      "f1": 0.43478260869565216,
      "precision": 0.6521739130434783,
      "recall": 0.32608695652173914,
      "balanced_accuracy": 0.6472954467648065,
      "roc_auc": 0.8928449161246148,
      "pr_auc": 0.6062631643618916
    }
  },
  {
    "run_id": "9364e416a38a:algorithm_random_forest__llm_proposed:v1",
    "algorithm_id": "algorithm_random_forest",
    "similarity": 0.98294,
    "metrics": {}
  },
  {
    "run_id": "9364e416a38a:algorithm_gradient_boosting__llm_proposed:v1",
    "algorithm_id": "algorithm_gradient_boosting",
    "similarity": 0.98294,
    "metrics": {
      "accuracy": 0.8326996197718631,
      "f1": 0.4358974358974359,
      "precision": 0.6538461538461539,
      "recall": 0.3269230769230769,
      "balanced_accuracy": 0.642134524243529,
      "roc_auc": 0.8554502369668247,
      "pr_auc": 0.5702531815904084
    }
  },
  {
    "run_id": "9364e416a38a",
    "algorithm_id": "algorithm_logistic_regression",
    "similarity": 0.98294,
    "metrics": {
      "accuracy": 0.8365019011406845,
      "f1": 0.45569620253164556,
      "precision": 0.6666666666666666,
      "recall": 0.34615384615384615,
      "balanced_accuracy": 0.6517499088589136,
      "roc_auc": 0.8655668975574189,
      "pr_auc": 0.6235057596449458
    }
  }
]
```

## 知识写回与版本

```json
{
  "workflow_run_id": "2b2a3d8a9bac",
  "validation_run_ids": [
    "2b2a3d8a9bac"
  ],
  "algorithm_version_ids": [
    "2b2a3d8a9bac_algorithm_logistic_regression__llm_proposed_v1"
  ],
  "failure_experience_ids": [
    "failure_2b2a3d8a9bac_algorithm_logistic_regression__llm_proposed_v1"
  ],
  "repair_experience_ids": [
    "repair_2b2a3d8a9bac_algorithm_logistic_regression__llm_proposed_v1"
  ],
  "capability_id": "capability_fe3d4b64611b254d"
}
```

## 真实调用计量

```json
[
  {
    "call_id": "llm_0001",
    "purpose": "requirement",
    "timestamp": "2026-09-07T16:56:20.534207+00:00",
    "prompt_sha256": "fe1d78298d1676e8bd257fbe1e44797b572d39453a19c706c6f1f01cbb665be1",
    "prompt_chars": 3950,
    "status": "ok",
    "response_sha256": "114db15ee2e37af1226bb3cc2cab4ef5d5f7f09bcf9f132ee2b3191835ed7162",
    "provider": "local_openai_compatible",
    "model": "Qwen2.5-14B-Instruct",
    "latency_ms": 32602.49,
    "token_usage": {
      "prompt_tokens": 1075,
      "completion_tokens": 519,
      "total_tokens": 1594
    },
    "retry_count": 1,
    "generation": {
      "purpose": "requirement",
      "max_new_tokens": 2400,
      "requested_max_new_tokens": 2400,
      "budget_adjustments": [],
      "finish_reason": "stop",
      "truncated": false,
      "eos_reached": true,
      "structured_transport": "json_object",
      "application_schema_validation": true
    }
  },
  {
    "call_id": "llm_0002",
    "purpose": "planning",
    "timestamp": "2026-09-07T16:57:57.414072+00:00",
    "prompt_sha256": "6632bcfc36fd41d6040bf4e5f359f21f9a970e1eb8e705450203cfd366ed7ff7",
    "prompt_chars": 22629,
    "status": "ok",
    "response_sha256": "ba7d5e3acb816c679547e8b1cc181dc6ea9b7f06259d5158ed6d17c5f4d7b40a",
    "provider": "local_openai_compatible",
    "model": "Qwen2.5-14B-Instruct",
    "latency_ms": 34266.35,
    "token_usage": {
      "prompt_tokens": 8769,
      "completion_tokens": 455,
      "total_tokens": 9224
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
      "structured_transport": "json_object",
      "application_schema_validation": true
    }
  },
  {
    "call_id": "llm_0003",
    "purpose": "code_generation",
    "timestamp": "2026-09-07T16:58:31.686771+00:00",
    "prompt_sha256": "ea66ecf06db7311bdcd8bf081866e682fb163bc65ae9b29a2bdb76d481e05ea1",
    "prompt_chars": 27889,
    "status": "ok",
    "response_sha256": "54fe4b9487776c218f7c0d1a0abc2e265401ea9a05ee353306934a3d83cada25",
    "provider": "local_openai_compatible",
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "latency_ms": 78656.97,
    "token_usage": {
      "prompt_tokens": 9853,
      "completion_tokens": 803,
      "total_tokens": 10656
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
    "purpose": "critique",
    "timestamp": "2026-09-07T16:59:50.717439+00:00",
    "prompt_sha256": "10ac939c8fb4e9efd1fd00be5d5024ffaa05850c37728955ad869f634ef3b831",
    "prompt_chars": 53702,
    "status": "ok",
    "response_sha256": "de0402c201921fb0760a28c76f39cc05fbc93db845c2748bfd837a8e79a15f4e",
    "provider": "local_openai_compatible",
    "model": "Qwen2.5-14B-Instruct",
    "latency_ms": 13321.04,
    "token_usage": {
      "prompt_tokens": 17006,
      "completion_tokens": 93,
      "total_tokens": 17099
    },
    "retry_count": 0,
    "generation": {
      "purpose": "critique",
      "max_new_tokens": 1024,
      "requested_max_new_tokens": 1024,
      "budget_adjustments": [],
      "finish_reason": "stop",
      "truncated": false,
      "eos_reached": true,
      "structured_transport": "json_object",
      "application_schema_validation": true
    }
  },
  {
    "call_id": "llm_0005",
    "purpose": "repair",
    "timestamp": "2026-09-07T17:00:04.042950+00:00",
    "prompt_sha256": "4d35fa3893c510d1c0c462b033a2d6c380d4b8d5e8a9df14bdb9868a2b87ba55",
    "prompt_chars": 64964,
    "status": "error",
    "error": "RuntimeError: BadRequestError: Error code: 400 - {'error': {'message': \"You passed 10243 input tokens and requested 6142 output tokens. However, the model's context length is only 16384 tokens, resulting in a maximum input length of 10242 tokens. Please reduce the length of the input prompt. (parameter=input_tokens, value=10243)\", 'type': 'BadRequestError', 'param': 'input_tokens', 'code': 400}}",
    "provider": "local_openai_compatible",
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "latency_ms": 257.58,
    "token_usage": {},
    "retry_count": 2,
    "generation": {}
  },
  {
    "call_id": "llm_0006",
    "purpose": "repair",
    "timestamp": "2026-09-07T17:00:04.302490+00:00",
    "prompt_sha256": "2250d4dfadbe98cb276be29043886a39e2db1268535d98166db3448d28ef1415",
    "prompt_chars": 65406,
    "status": "error",
    "error": "RuntimeError: BadRequestError: Error code: 400 - {'error': {'message': \"You passed 10243 input tokens and requested 6142 output tokens. However, the model's context length is only 16384 tokens, resulting in a maximum input length of 10242 tokens. Please reduce the length of the input prompt. (parameter=input_tokens, value=10243)\", 'type': 'BadRequestError', 'param': 'input_tokens', 'code': 400}}",
    "provider": "local_openai_compatible",
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "latency_ms": 198.62,
    "token_usage": {},
    "retry_count": 2,
    "generation": {}
  },
  {
    "call_id": "llm_0007",
    "purpose": "explanation",
    "timestamp": "2026-09-07T17:00:09.880703+00:00",
    "prompt_sha256": "167417d69050d072df3e4de4fd767c7fb874eca2e93d01611b762446e30f00dd",
    "prompt_chars": 24831,
    "status": "error",
    "error": "RuntimeError: APIConnectionError: Connection error.",
    "provider": "local_openai_compatible",
    "model": "Qwen2.5-14B-Instruct",
    "latency_ms": 4296.22,
    "token_usage": {},
    "retry_count": 0,
    "generation": {}
  },
  {
    "call_id": "llm_0008",
    "purpose": "explanation",
    "timestamp": "2026-09-07T17:00:14.179178+00:00",
    "prompt_sha256": "affa166f36f225f0fc8ae220b3a6a6ea2d5725db479435067dedc93cd3fa9755",
    "prompt_chars": 24906,
    "status": "error",
    "error": "RuntimeError: APIConnectionError: Connection error.",
    "provider": "local_openai_compatible",
    "model": "Qwen2.5-14B-Instruct",
    "latency_ms": 2.53,
    "token_usage": {},
    "retry_count": 0,
    "generation": {}
  }
]
```

## 修复与不可变代码版本

- 2b2a3d8a9bac_algorithm_logistic_regression__llm_proposed_v1: failed, SHA256=ef7b1b8b1ed5c6811b1689b4b260990980e21e5316d16a92cd8b6caf4d67942d

```json
[
  {
    "round": 1,
    "before_sha256": "ef7b1b8b1ed5c6811b1689b4b260990980e21e5316d16a92cd8b6caf4d67942d",
    "changes": [
      "no accepted LLM revision; source unchanged"
    ],
    "retrieved_experience_ids": [
      "failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v4",
      "failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v3",
      "failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v2",
      "failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v1",
      "failure_9364e416a38a_algorithm_random_forest__llm_proposed_v1"
    ],
    "error": "{\"validation\": {\"status\": \"failed\", \"algorithm\": \"Logistic Regression\", \"checks\": {\"static_safety\": {\"passed\": false, \"category\": \"interface\", \"message\": \"missing function: predict\"}, \"functional\": {\"passed\": false, \"message\": \"missing function: predict\"}, \"runtime_budget\": {\"passed\": true, \"runtime_seconds\": 0.28384560346603394, \"timeout_seconds\": 90}}, \"metrics\": {}, \"runtime_seconds\": 0.28384560346603394, \"errors\": [\"ValueError: missing function: predict\"], \"warnings\": [], \"stdout\": \"\", \"stderr\": \"\", \"repair_round\": 0, \"task_type\": \"binary_classification\", \"dataset_profile\": {}, \"resource_usage\": {\"code_hash\": \"ef7b1b8b1ed5c6811b1689b4b260990980e21e5316d16a92cd8b6caf4d67942d\"}, \"failure_type\": \"interface_failure\", \"root_cause\": \"ValueError: missing function: predict\"}, \"diagnosis\": {\"failure_type\": \"interface_failure\", \"root_cause\": \"missing function: predict\", \"triggering_condition\": \"The provided code does not implement the required function 'predict'.\", \"repair_strategy\": \"Implement the 'predict' function to ensure the code meets the interface requirements.\", \"reusable_lesson\": \"Always ensure that all required functions are implemented when developing machine learning models to avoid interface failures during validation.\", \"provider\": \"local_openai_compatible\", \"status\": \"ok\", \"observed_failure_type\": \"interface_failure\", \"observed_error\": \"ValueError: missing function: predict\", \"retrieved_experience_ids\": [\"failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v4\", \"failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v3\", \"failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v2\", \"failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v1\", \"failure_9364e416a38a_algorithm_random_forest__llm_proposed_v1\"]}, \"metric_thresholds\": {\"roc_auc\": 0.8}, \"resource_constraints\": {}}",
    "attempts": [
      {
        "attempt": 1,
        "status": "rejected",
        "error": "LLMInvocationError: RuntimeError: BadRequestError: Error code: 400 - {'error': {'message': \"You passed 10243 input tokens and requested 6142 output tokens. However, the model's context length is only 16384 tokens, resulting in a maximum input length of 10242 tokens. Please reduce the length of the input prompt. (parameter=input_tokens, value=10243)\", 'type': 'BadRequestError', 'param': 'input_tokens', 'code': 400}}"
      },
      {
        "attempt": 2,
        "status": "rejected",
        "error": "LLMInvocationError: RuntimeError: BadRequestError: Error code: 400 - {'error': {'message': \"You passed 10243 input tokens and requested 6142 output tokens. However, the model's context length is only 16384 tokens, resulting in a maximum input length of 10242 tokens. Please reduce the length of the input prompt. (parameter=input_tokens, value=10243)\", 'type': 'BadRequestError', 'param': 'input_tokens', 'code': 400}}"
      }
    ],
    "provider": "openai",
    "status": "repair_rejected",
    "after_sha256": "ef7b1b8b1ed5c6811b1689b4b260990980e21e5316d16a92cd8b6caf4d67942d",
    "diagnosis": {
      "failure_type": "interface_failure",
      "root_cause": "missing function: predict",
      "triggering_condition": "The provided code does not implement the required function 'predict'.",
      "repair_strategy": "Implement the 'predict' function to ensure the code meets the interface requirements.",
      "reusable_lesson": "Always ensure that all required functions are implemented when developing machine learning models to avoid interface failures during validation.",
      "provider": "local_openai_compatible",
      "status": "ok",
      "observed_failure_type": "interface_failure",
      "observed_error": "ValueError: missing function: predict",
      "retrieved_experience_ids": [
        "failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v4",
        "failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v3",
        "failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v2",
        "failure_6bde3f2c39b8_algorithm_random_forest__llm_proposed_v1",
        "failure_9364e416a38a_algorithm_random_forest__llm_proposed_v1"
      ]
    },
    "from_version": "2b2a3d8a9bac_algorithm_logistic_regression__llm_proposed_v1"
  }
]
```

## 工作流轨迹

- [RequirementAgent] started
- [RequirementAgent] completed
- [KnowledgeExtractionAgent] started
- [KnowledgeExtractionAgent] completed
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
- [DemoFaultInjection] applied
- [Sandbox] started
- [ValidatorAgent] started
- [ValidatorAgent] completed
- [ValidatorAgent] failed
- [CriticAgent] started
- [CriticAgent] completed
- [CriticAgent] ok
- [RepairAgent] started
- [RepairAgent] completed
- [RepairAgent] repair_rejected
- [CuratorAgent] started
- [CuratorAgent] completed
- [ExplanationAgent] started
- [ExplanationAgent] completed

限制：当前数据与切分上的实验结果不能保证生产效果；原型沙箱不是对抗恶意代码的生产安全边界。
