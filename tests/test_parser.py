from app.agents.parser_agent import ParserAgent


def test_parser_extracts_churn_contract(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("age,region,login_count_30d,churn\n20,east,4,1\n30,west,10,0\n", encoding="utf-8")
    spec = ParserAgent().run("根据年龄、地区和登录频率预测客户是否流失，要求 ROC-AUC 不低于 0.8。", str(path))
    assert spec.task_type == "binary_classification"
    assert spec.target_column == "churn"
    assert spec.metric_thresholds["roc_auc"] == 0.8
    assert spec.feature_columns == ["age", "region", "login_count_30d"]

