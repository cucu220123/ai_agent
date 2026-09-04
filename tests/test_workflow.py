from app.config import Settings
from app.workflow import AlgorithmFactoryWorkflow
from scripts.generate_demo_data import generate


def test_end_to_end(tmp_path):
    data_path = generate(tmp_path / "churn.csv", n_rows=240)
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        generated_dir=tmp_path / "generated",
        reports_dir=tmp_path / "reports",
        knowledge_db=tmp_path / "knowledge.sqlite",
        graphml_path=tmp_path / "knowledge.graphml",
        llm_provider="mock",
        validation_timeout_seconds=30,
        max_repair_rounds=1,
    )
    result = AlgorithmFactoryWorkflow(settings).run("根据客户登录频率、消费金额和投诉次数预测是否流失，ROC-AUC 不低于 0.65。", data_path)
    assert result.validation is not None
    assert result.validation.status == "passed", result.validation.errors
    assert result.report_markdown
    assert result.llm_trace["structured_advice"]["status"] in {"ok", "fallback"}
    assert result.candidate_results
