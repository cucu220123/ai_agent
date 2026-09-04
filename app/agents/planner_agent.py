from __future__ import annotations

from app.models import AlgorithmPlan, CapabilitySpec, KnowledgeContext


class PlannerAgent:
    _defaults = {
        "logistic_regression": ("Logistic Regression", ["median imputation", "most-frequent imputation", "one-hot encoding", "standard scaling"], {"C": 1.0, "max_iter": 500}),
        "random_forest": ("Random Forest", ["median imputation", "most-frequent imputation", "one-hot encoding"], {"n_estimators": 180, "max_depth": 8, "random_state": 42}),
        "gradient_boosting": ("Gradient Boosting", ["median imputation", "most-frequent imputation", "one-hot encoding"], {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}),
    }

    def run(self, spec: CapabilitySpec, knowledge: KnowledgeContext) -> list[AlgorithmPlan]:
        historical = {}
        for item in knowledge.algorithms:
            key = item.get("id", "").replace("algorithm_", "")
            historical[key] = item.get("historical_metrics", {})
        names = spec.candidate_algorithms or list(self._defaults)
        plans = []
        for priority, key in enumerate(names):
            if key not in self._defaults:
                continue
            name, preprocessing, params = self._defaults[key]
            expected = historical.get(key, {})
            rationale = {
                "logistic_regression": "类别和数值特征经过统一预处理后，逻辑回归提供高可解释性和低资源消耗。",
                "random_forest": "随机森林对非线性关系和特征尺度不敏感，适合混合类型客户行为特征。",
                "gradient_boosting": "梯度提升通常具有较高的排序能力，作为性能优先候选方案。",
            }[key]
            if "prefer_interpretable" in spec.constraints and key == "logistic_regression":
                priority -= 2
            plans.append(AlgorithmPlan(
                algorithm_id=f"algorithm_{key}", algorithm_name=name, rationale=rationale,
                preprocessing=preprocessing, hyperparameters=params, expected_metrics=expected,
                priority=priority,
            ))
        return sorted(plans, key=lambda p: (p.priority, -p.expected_metrics.get("roc_auc", 0.0)))

