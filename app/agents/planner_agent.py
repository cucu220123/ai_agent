from __future__ import annotations

from app.models import AlgorithmPlan, CapabilitySpec, KnowledgeContext
from app.plugins.registry import DEFAULT_REGISTRY


class PlannerAgent:
    _defaults = {
        "logistic_regression": ("Logistic Regression", ["median imputation", "most-frequent imputation", "one-hot encoding", "standard scaling"], {"C": 1.0, "max_iter": 500}),
        "random_forest": ("Random Forest", ["median imputation", "most-frequent imputation", "one-hot encoding"], {"n_estimators": 180, "max_depth": 8, "random_state": 42}),
        "gradient_boosting": ("Gradient Boosting", ["median imputation", "most-frequent imputation", "one-hot encoding"], {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}),
        "random_forest_regressor": ("Random Forest Regressor", ["median imputation", "most-frequent imputation", "one-hot encoding"], {"n_estimators": 160, "max_depth": 10, "random_state": 42}),
        "isolation_forest": ("Isolation Forest", ["median imputation", "standard scaling"], {"n_estimators": 160, "contamination": "auto", "random_state": 42}),
        "tfidf_logistic_regression": ("TF-IDF Logistic Regression", ["text fillna", "TF-IDF unigram/bigram"], {"max_features": 5000, "max_iter": 500}),
    }

    def run(self, spec: CapabilitySpec, knowledge: KnowledgeContext) -> list[AlgorithmPlan]:
        historical = {item.get("id", "").replace("algorithm_", ""): item.get("historical_metrics", {}) for item in knowledge.algorithms}
        supported = {p.id for p in DEFAULT_REGISTRY.algorithms_for(spec.task_type)}
        names = [name for name in (spec.candidate_algorithms or list(self._defaults)) if name in supported]
        plans = []
        for priority, key in enumerate(names):
            name, preprocessing, params = self._defaults[key]
            expected = historical.get(key, {})
            rationale = {
                "logistic_regression": "类别和数值特征经过统一预处理后，逻辑回归提供高可解释性和低资源消耗。",
                "random_forest": "随机森林对非线性关系和特征尺度不敏感，适合混合类型客户行为特征。",
                "gradient_boosting": "梯度提升通常具有较高的排序能力，作为性能优先候选方案。",
                "random_forest_regressor": "随机森林回归对非线性数值关系鲁棒，作为无需缩放的稳健回归方案。",
                "isolation_forest": "Isolation Forest 通过随机划分识别稀有样本，适合无监督异常检测。",
                "tfidf_logistic_regression": "TF-IDF 与逻辑回归组合轻量、稳定，适合作为文本分类和跨场景迁移基线。",
            }[key]
            if "prefer_interpretable" in spec.constraints and key == "logistic_regression":
                priority -= 2
            plans.append(AlgorithmPlan(f"algorithm_{key}", name, rationale, preprocessing, params, expected, priority))
        return sorted(plans, key=lambda p: (p.priority, -p.expected_metrics.get("roc_auc", 0.0)))
