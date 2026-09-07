import pytest
from app.agents.generator_agent import GeneratorAgent
from app.agents.planner_agent import PlannerAgent
from app.models import CapabilitySpec, KnowledgeContext
from app.validation.runner import ValidationRunner
from app.validation.checks import static_check_source
from scripts.generate_demo_data import generate


def fixture_code(tmp_path):
    spec = CapabilitySpec(raw_description="integrity", metric_thresholds={})
    plan = PlannerAgent().run(spec, KnowledgeContext())[0]
    return GeneratorAgent().run(tmp_path / "generated", spec, plan), spec


@pytest.mark.parametrize("payload", ["pd.read_csv('/etc/passwd')", "np.load('secret.npy')", "x.__class__.__base__", "getattr(x, '__globals__')", "f = open", "__import__('socket')"])
def test_io_and_reflection_are_blocked(payload):
    code = "import pandas as pd\nimport numpy as np\n" + payload
    gate = static_check_source(code)
    assert not gate["passed"] and gate["category"] == "security"


def test_generated_evaluate_cannot_fabricate_auc(tmp_path):
    path, spec = fixture_code(tmp_path)
    code = path.read_text()
    code = code.replace('"roc_auc": float(roc_auc_score(y_true, pred["probability"]))', '"roc_auc": 1.0')
    path.write_text(code)
    data = generate(tmp_path / "data.csv", n_rows=180)
    result = ValidationRunner(30).run(path, data, spec, "fabricated")
    assert result.status == "failed" and result.failure_type == "metric_integrity_failure", result.errors
    assert result.metrics["roc_auc"] < 1
    assert not result.checks["metric_integrity"]["passed"]


def test_missing_dataset_returns_report_instead_of_unbound_local(tmp_path):
    path, spec = fixture_code(tmp_path)
    result = ValidationRunner(30).run(path, tmp_path / "missing.csv", spec, "missing")
    assert result.status == "failed"
    assert "FileNotFoundError" in str(result.errors)


def test_duplicate_target_leakage_rejected(tmp_path):
    import pandas as pd
    path, spec = fixture_code(tmp_path)
    data = generate(tmp_path / "data.csv", n_rows=100)
    frame = pd.read_csv(data)
    frame["future_label_copy"] = frame.churn
    frame.to_csv(data, index=False)
    result = ValidationRunner(30).run(path, data, spec, "leaky")
    assert result.status == "failed" and not result.checks["target_leakage"]["passed"]


def test_real_resource_limits_and_optional_cross_validation(tmp_path):
    path, spec = fixture_code(tmp_path)
    data = generate(tmp_path / "data.csv", n_rows=180)
    result = ValidationRunner(30, cv_folds=3).run(path, data, spec, "valid")
    assert result.status == "passed", result.errors
    assert result.checks["isolated_execution"]["environment_sanitized"]
    assert result.resource_usage["limits"]["applied"]
    assert result.resource_usage["sandbox"]["network_audit_blocked"]
    assert len(result.checks["cross_validation"]["folds"]) == 3


def test_unstable_predictions_fail(tmp_path):
    path, spec = fixture_code(tmp_path)
    code = path.read_text().replace('probability = model.predict_proba(test_df)[:, 1]', 'probability = np.random.default_rng().uniform(0.1, 0.9, len(test_df))')
    # Keep evaluate and predict_proba deterministic for each prediction invocation by
    # instead vary the fitted model between repeated train calls.
    code = path.read_text().replace('C=1.0', 'C=float(np.random.default_rng().choice([0.00001, 1000.0]))')
    # A module counter makes failure reproducible rather than probabilistic.
    code = code.replace('ALGORITHM_NAME =', 'TRAIN_COUNT = 0\nALGORITHM_NAME =')
    code = code.replace('    model = _build_pipeline(train_df, target_col, config)', '    global TRAIN_COUNT\n    TRAIN_COUNT += 1\n    model = _build_pipeline(train_df, target_col, config)\n    model.set_params(model__C=0.00001 if TRAIN_COUNT % 2 else 1000.0)')
    path.write_text(code)
    data = generate(tmp_path / "data.csv", n_rows=180)
    result = ValidationRunner(30).run(path, data, spec, "unstable")
    assert result.status == "failed" and result.failure_type == "instability", result.errors
