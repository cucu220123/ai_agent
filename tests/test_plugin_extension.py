from app.agents.planner_agent import PlannerAgent
from app.models import CapabilitySpec, KnowledgeContext
from app.plugins.registry import AlgorithmPlugin, DEFAULT_REGISTRY
from app.search.beam import BeamSearchPlanner


def test_toy_plugin_enters_planner_and_beam_without_core_changes():
    plugin_id = "toy_classifier"
    DEFAULT_REGISTRY.register_algorithm(AlgorithmPlugin(plugin_id, "Toy Classifier", ["binary_classification"], "test-only plugin", "low", {"alpha": 1.0}, ["none"], {"alpha": [0.1, 1.0]}, ["f1"], {"protocol": "tabular_binary"}))
    spec = CapabilitySpec(raw_description="toy", candidate_algorithms=[plugin_id], task_type="binary_classification")
    plans = PlannerAgent().run(spec, KnowledgeContext())
    assert plans and plans[0].algorithm_name == "Toy Classifier"
    states, trace = BeamSearchPlanner().search(plans, spec, beam_width=2)
    assert states
    assert trace.expanded >= 2
