import numpy as np
import pytest
from app.agents.protocol import AgentRuntime
from app.agents.generator_agent import GeneratorAgent
from app.agents.planner_agent import PlannerAgent
from app.plugins.registry import DEFAULT_REGISTRY, AlgorithmPlugin
from app.metrics.registry import METRIC_REGISTRY, MetricDefinition
from app.models import CapabilitySpec, KnowledgeContext
from app.validation.runner import ValidationRunner
from scripts.generate_demo_data import generate


def test_agent_cannot_invoke_another_roles_tool():
    runtime = AgentRuntime([])
    called = []
    with pytest.raises(PermissionError):
        runtime.call("RequirementAgent", "curate", lambda: called.append(True))
    assert not called
    assert runtime.call("RequirementAgent", "understand", lambda: "parsed") == "parsed"
    assert runtime.events[-1]["status"] == "completed"


def test_registered_renderer_and_parent_metric_execute_without_core_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(DEFAULT_REGISTRY, "algorithms", dict(DEFAULT_REGISTRY.algorithms))
    monkeypatch.setattr(METRIC_REGISTRY, "metrics", dict(METRIC_REGISTRY.metrics))
    monkeypatch.setattr(METRIC_REGISTRY, "evaluators", dict(METRIC_REGISTRY.evaluators))
    def renderer(spec, plan):
        return '''import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
def train(train_df,target_col,config=None):
    model = DummyClassifier(strategy="prior")
    model.fit(train_df.drop(columns=[target_col]), train_df[target_col])
    return model
def predict(model,test_df):
    p=model.predict_proba(test_df)[:,1]
    return pd.DataFrame({"prediction":model.predict(test_df),"probability":p})
def evaluate(model,test_df,target_col):
    return {"roc_auc":float(roc_auc_score(test_df[target_col],predict(model,test_df.drop(columns=[target_col]))["probability"]))}
'''
    plugin = AlgorithmPlugin("fixture_dummy", "Fixture Dummy", ["binary_classification"], "independent plugin", renderer=renderer)
    DEFAULT_REGISTRY.register_algorithm(plugin)
    METRIC_REGISTRY.register(MetricDefinition("specificity", True, ("binary_classification",)), lambda y, pred: np.mean(np.asarray(pred["prediction"])[y == 0] == 0))
    spec = CapabilitySpec(raw_description="plugin", candidate_algorithms=[plugin.id], metrics=["specificity"], metric_thresholds={"specificity": .99})
    plan = PlannerAgent().run(spec, KnowledgeContext())[0]
    code = GeneratorAgent().run(tmp_path / "generated", spec, plan)
    result = ValidationRunner(30).run(code, generate(tmp_path / "data.csv", n_rows=180), spec, plugin.name)
    assert result.status == "passed", result.errors
    assert result.metrics["specificity"] == 1
    assert "DummyClassifier" in code.read_text() and "GradientBoostingClassifier" not in code.read_text()
