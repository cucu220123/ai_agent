from app.agents.parser_agent import ParserAgent
from app.plugins.registry import DEFAULT_REGISTRY


def test_task_plugins_are_registered():
    assert {"binary_classification", "regression", "anomaly_detection"}.issubset(DEFAULT_REGISTRY.tasks)
    assert DEFAULT_REGISTRY.algorithms_for("anomaly_detection")


def test_parser_anomaly_task():
    spec = ParserAgent().run("对设备传感器数据进行异常检测")
    assert spec.task_type == "anomaly_detection"
    assert spec.target_column == ""

