from app.config import Settings
from app.workflow import AlgorithmFactoryWorkflow
from scripts.generate_demo_data import generate


def _settings(tmp_path):
    return Settings(project_root=tmp_path, data_dir=tmp_path / "data", generated_dir=tmp_path / "generated", reports_dir=tmp_path / "reports", knowledge_db=tmp_path / "knowledge.sqlite", graphml_path=tmp_path / "knowledge.graphml", llm_provider="mock", validation_timeout_seconds=30, max_repair_rounds=1)


def test_regression_workflow(tmp_path):
    data = tmp_path / "regression.csv"
    data.write_text("age,visits,spend\n20,2,100\n22,3,120\n25,4,150\n28,5,190\n30,6,220\n32,7,250\n35,8,290\n38,9,330\n40,10,365\n42,11,390\n45,12,430\n48,13,470\n50,14,500\n52,15,540\n55,16,580\n58,17,610\n60,18,650\n62,19,690\n65,20,720\n68,21,760\n", encoding="utf-8")
    result = AlgorithmFactoryWorkflow(_settings(tmp_path)).run("根据年龄、访问次数预测消费金额，做回归预测，要求 MAE 不高于 100", data)
    assert result.validation.status == "passed", result.validation.errors
    assert result.validation.metrics["mae"] <= 100


def test_anomaly_workflow(tmp_path):
    data = tmp_path / "anomaly.csv"
    generate_path = __import__("pathlib").Path(__file__).parents[1] / "data" / "anomaly_demo.csv"
    data.write_text(generate_path.read_text(encoding="utf-8"), encoding="utf-8")
    result = AlgorithmFactoryWorkflow(_settings(tmp_path)).run("对设备温度、振动和压力数据进行异常检测", data)
    assert result.validation.status == "passed", result.validation.errors
    assert "anomaly_rate" in result.validation.metrics


def test_text_workflow(tmp_path):
    data = tmp_path / "text.csv"
    source = __import__("pathlib").Path(__file__).parents[1] / "data" / "text_demo.csv"
    data.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    result = AlgorithmFactoryWorkflow(_settings(tmp_path)).run("对文本评论进行文本分类，预测正面或负面", data)
    assert result.validation.status == "passed", result.validation.errors
    assert "f1" in result.validation.metrics
