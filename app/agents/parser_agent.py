from __future__ import annotations

import re

from app.models import CapabilitySpec


class ParserAgent:
    """Extract a stable task contract from Chinese or English descriptions."""

    _known_features = {
        "age": ["年龄", "age"],
        "region": ["地区", "区域", "region"],
        "login_count_30d": ["登录频率", "登录次数", "活跃次数", "login_count", "login frequency"],
        "total_spend": ["消费金额", "历史消费", "总消费", "total_spend", "spend"],
        "complaint_count": ["投诉次数", "投诉", "complaint_count", "complaints"],
        "membership_level": ["会员等级", "会员级别", "membership_level", "membership"],
        "tenure_months": ["使用时长", "在网时长", "tenure"],
        "last_login_days": ["距上次登录", "last_login_days"],
    }

    def run(self, description: str, dataset_path: str | None = None) -> CapabilitySpec:
        text = description.strip()
        lowered = text.lower()
        if not text:
            raise ValueError("能力描述不能为空")

        task_type = "binary_classification"
        if any(word in lowered for word in ("异常检测", "anomaly detection", "outlier")):
            task_type = "anomaly_detection"
        elif any(word in lowered for word in ("文本分类", "text classification")):
            task_type = "text_classification"
        elif any(word in lowered for word in ("回归", "regression", "预测价格", "预测销量")):
            task_type = "regression"

        target = "churn"
        target_match = re.search(r"(?:目标|标签|target)(?:列|字段)?\s*[为是:]?\s*[`'\"]?([A-Za-z_][\w-]*)", text, re.I)
        if target_match:
            target = target_match.group(1)
        if "流失" in text or "churn" in lowered:
            target = "churn"

        features: list[str] = []
        for canonical, aliases in self._known_features.items():
            if any(alias.lower() in lowered for alias in aliases):
                features.append(canonical)
        if dataset_path:
            try:
                import pandas as pd

                columns = list(pd.read_csv(dataset_path, nrows=0).columns)
                if target in columns:
                    features = [c for c in columns if c != target]
            except Exception:
                pass

        metrics = ["roc_auc", "f1", "precision", "recall"] if task_type == "binary_classification" else ["rmse", "mae"]
        if "auc" in lowered or "roc" in lowered:
            metrics = ["roc_auc", *[m for m in metrics if m != "roc_auc"]]
        if "f1" in lowered:
            metrics = ["f1", *[m for m in metrics if m != "f1"]]

        thresholds: dict[str, float] = {}
        for metric_alias, metric in (("roc[- ]?auc|auc", "roc_auc"), ("f1", "f1"), ("准确率|accuracy", "accuracy"), ("召回率|recall", "recall")):
            match = re.search(rf"(?:{metric_alias})\s*(?:不低于|至少|>=|大于等于|不少于|为)?\s*([01](?:\.\d+)?)", lowered, re.I)
            if match:
                thresholds[metric] = float(match.group(1))
        if not thresholds and task_type == "binary_classification":
            thresholds = {"roc_auc": 0.75}

        candidates: list[str] = []
        mappings = {
            "logistic_regression": ["逻辑回归", "logistic"],
            "random_forest": ["随机森林", "random forest"],
            "gradient_boosting": ["梯度提升", "gradient boosting", "gbdt"],
        }
        for name, aliases in mappings.items():
            if any(alias in lowered for alias in aliases):
                candidates.append(name)
        if not candidates and task_type == "binary_classification":
            candidates = ["logistic_regression", "random_forest", "gradient_boosting"]

        constraints = []
        if "可解释" in text or "explain" in lowered:
            constraints.append("prefer_interpretable")
        if "低延迟" in text or "low latency" in lowered:
            constraints.append("low_latency")
        if "概率" in text or "probability" in lowered:
            constraints.append("output_probability")
        return CapabilitySpec(
            raw_description=text,
            capability_name="客户流失预测" if target == "churn" else "自定义算法能力",
            task_type=task_type,
            target_column=target,
            feature_columns=features,
            metrics=metrics,
            metric_thresholds=thresholds,
            constraints=constraints,
            candidate_algorithms=candidates,
            dataset_path=dataset_path,
        )

