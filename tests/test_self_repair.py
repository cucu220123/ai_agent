from app.experience.self_repair import run_self_repair_demo


def test_self_repair_demo_passes(tmp_path):
    result = run_self_repair_demo(tmp_path / "repair")
    assert result["first_validation"]["status"] == "failed"
    assert result["second_validation"]["status"] == "passed"


def test_repair_line_array_preserves_python_layout():
    import pytest
    import ast
    from app.agents.repair_agent import RepairContract
    lines = ['import pandas as pd', '', 'def predict(model, test_df):', '    return pd.DataFrame({"prediction": model.predict(test_df)})']
    repair = RepairContract(failure_diagnosis='missing API', repair_strategy='Restore the expected function', code_lines=lines)
    ast.parse(repair.revised_code)
    assert repair.revised_code.splitlines() == lines
    with pytest.raises(ValueError, match='not both'):
        RepairContract(failure_diagnosis='missing API', repair_strategy='Restore the expected function', code_lines=lines, revised_code=repair.revised_code)
