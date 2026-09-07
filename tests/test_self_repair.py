from app.experience.self_repair import run_self_repair_demo


def test_self_repair_demo_passes(tmp_path):
    result = run_self_repair_demo(tmp_path / "repair")
    assert result["first_validation"]["status"] == "failed"
    assert result["second_validation"]["status"] == "passed"
