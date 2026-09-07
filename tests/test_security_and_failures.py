from pathlib import Path

from app.models import CapabilitySpec
from app.validation.checks import static_check
from app.validation.isolate import run_isolated
from app.validation.runner import ValidationRunner
from scripts.generate_demo_data import generate


def _write(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_malicious_import_blocked(tmp_path):
    path = _write(tmp_path / "bad.py", "import os\ndef train(a,b,config=None): pass\ndef predict(a,b): pass\ndef evaluate(a,b,c): pass\n")
    result = static_check(path)
    assert not result["passed"]
    assert "disallowed import" in result["message"]


def test_interface_failure_blocked(tmp_path):
    path = _write(tmp_path / "bad.py", "def train(a,b,config=None): pass\ndef evaluate(a,b,c): pass\n")
    result = static_check(path)
    assert not result["passed"]
    assert "missing function: predict" in result["message"]


def test_timeout_is_reported(tmp_path):
    data = generate(tmp_path / "data.csv", n_rows=60)
    path = _write(tmp_path / "slow.py", """import pandas as pd
def train(train_df,target_col,config=None):
    while True: pass
def predict(model,test_df): return pd.DataFrame({'prediction':[0]*len(test_df),'probability':[0.5]*len(test_df)})
def evaluate(model,test_df,target_col): return {'roc_auc':0.5}
""")
    result = run_isolated(path, data, "churn", 1)
    assert not result["passed"]
    assert result["timeout"] or "exit" in result["message"]


def test_metric_failure_is_explicit(tmp_path):
    data = generate(tmp_path / "data.csv", n_rows=180)
    source = tmp_path / "algorithm.py"
    if not source.exists():
        from app.agents.generator_agent import GeneratorAgent
        from app.agents.planner_agent import PlannerAgent
        from app.models import KnowledgeContext
        from app.generation.templates import render_algorithm
        plan = PlannerAgent().run(CapabilitySpec(raw_description="x"), KnowledgeContext())[0]
        source = tmp_path / "algorithm.py"
        source.write_text(render_algorithm(CapabilitySpec(raw_description="x"), plan), encoding="utf-8")
    result = ValidationRunner(30).run(source, data, CapabilitySpec(raw_description="x", metric_thresholds={"roc_auc": 0.9999}), "test")
    assert result.status == "failed"
    assert any("roc_auc" in error for error in result.errors)
