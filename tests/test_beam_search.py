from app.agents.planner_agent import PlannerAgent
from app.generation.templates import render_algorithm
from app.models import CapabilitySpec, KnowledgeContext
from app.search.beam import BeamSearchPlanner


def test_beam_states_change_real_code_and_trace():
    spec = CapabilitySpec(raw_description="x", class_imbalance={"is_imbalanced": True})
    parents = PlannerAgent().run(spec, KnowledgeContext())
    selected, trace = BeamSearchPlanner().search(parents, spec, beam_width=3)
    assert trace.expanded > trace.beam_width
    assert trace.expansions and trace.pruned
    assert len({plan.base_algorithm_id for plan in selected}) == 3
    logistic_states, _ = BeamSearchPlanner().search([parents[0]], spec, beam_width=2)
    sources = [render_algorithm(spec, state) for state in logistic_states]
    assert len(set(sources)) == len(sources)

