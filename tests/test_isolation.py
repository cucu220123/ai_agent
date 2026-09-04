from pathlib import Path

from app.models import CapabilitySpec, KnowledgeContext
from app.agents.generator_agent import GeneratorAgent
from app.agents.planner_agent import PlannerAgent
from app.validation.isolate import run_isolated
from scripts.generate_demo_data import generate


def test_isolated_execution(tmp_path):
    data = generate(tmp_path / "data.csv", n_rows=180)
    spec = CapabilitySpec(raw_description="test", feature_columns=["age"])
    plan = PlannerAgent().run(spec, KnowledgeContext())[0]
    algorithm = GeneratorAgent().run(tmp_path / "generated", spec, plan)
    result = run_isolated(algorithm, data, "churn", 30)
    assert result["passed"], result
    assert result["metrics"]["roc_auc"] > 0.5

