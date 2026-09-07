from app.metrics.registry import METRIC_REGISTRY


def test_cross_task_metric_directions():
    assert METRIC_REGISTRY.ranking_score("binary_classification", {"roc_auc": 0.9}, 2, ["roc_auc"])[0] > METRIC_REGISTRY.ranking_score("binary_classification", {"roc_auc": 0.8}, 1, ["roc_auc"])[0]
    assert METRIC_REGISTRY.ranking_score("regression", {"mae": 10}, 2, ["mae"])[0] > METRIC_REGISTRY.ranking_score("regression", {"mae": 20}, 1, ["mae"])[0]
    assert METRIC_REGISTRY.ranking_score("text_classification", {"f1": 0.7}, 2, ["f1"])[0] > METRIC_REGISTRY.ranking_score("text_classification", {"f1": 0.6}, 1, ["f1"])[0]
    score, metric = METRIC_REGISTRY.ranking_score("anomaly_detection", {"anomaly_rate": 0.8}, 1.2, [], False)
    assert metric == "runtime_seconds"
    assert score == -1.2

