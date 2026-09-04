from pathlib import Path

from app.agents.generator_agent import GeneratorAgent
from app.agents.planner_agent import PlannerAgent
from app.models import CapabilitySpec, KnowledgeContext
from app.validation.checks import static_check


def test_generated_template_is_safe(tmp_path):
    spec = CapabilitySpec(raw_description="test", feature_columns=["age"])
    plan = PlannerAgent().run(spec, KnowledgeContext())[0]
    path = GeneratorAgent().run(tmp_path, spec, plan)
    check = static_check(path)
    assert check["passed"], check

