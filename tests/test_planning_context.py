from app.agents.planning_context import PlanningContextBuilder
from app.models import CapabilitySpec, KnowledgeContext


def test_planning_context_is_compressed_and_traced():
    knowledge = KnowledgeContext(
        graph_evidence={"nodes": [{"id": f"n{i}"} for i in range(100)], "serialized": {"candidate_algorithms": [{"algorithm_id": f"a{i}", "evidence_node_ids": [f"e{i}"], "payload": "x" * 1000} for i in range(20)]}},
        historical_cases=[{"run_id": f"r{i}", "payload": "y" * 1000} for i in range(20)],
        experiences=[{"id": f"f{i}", "kind": "failure", "payload": "z" * 1000} for i in range(20)],
        semantic_evidence=[{"source": {"id": f"s{i}", "text": "v" * 1000}} for i in range(20)],
    )
    context, trace = PlanningContextBuilder(max_chars=7000).build(CapabilitySpec(raw_description="x"), knowledge)
    assert trace["raw_retrieved_items"]["graph_nodes"] == 100
    assert trace["items_after_rerank"]["historical_runs"] <= 8
    assert trace["final_prompt_chars"] <= 7000
    assert trace["estimated_prompt_tokens"] > 0
    assert context["graph_candidates"]
