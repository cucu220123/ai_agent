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


def test_planner_only_normalizes_evidence_aliases_from_verified_run_metadata():
    import json
    from app.agents.advisor_agent import AdvisorAgent
    knowledge = KnowledgeContext(algorithms=[{'id': 'algorithm_logistic_regression', 'task_types': ['binary_classification']}], historical_cases=[{'run_id': 'Run_A', 'workflow_run_id': 'Run_A', 'candidate_id': 'algorithm_logistic_regression__c1', 'version_id': 'Run_A_algorithm_logistic_regression__c1_v1', 'task_type': 'binary_classification', 'metrics': {'roc_auc': 0.87}, 'similarity': 1.0}])
    class Model:
        def __init__(self, evidence): self.evidence = evidence
        def complete(self, *args, **kwargs):
            return json.dumps({'candidate_algorithms': ['logistic_regression'], 'preprocessing_recommendations': {'logistic_regression': ['scaling']}, 'metric_strategy': {}, 'constraint_analysis': [], 'evidence_ids': [self.evidence], 'algorithm_reasons': {'logistic_regression': 'similar measured run'}, 'risks': [], 'confidence': 0.8})
    alias = 'Run_A:algorithm_logistic_regression__c1:v1'
    advice, trace = AdvisorAgent(Model(alias), max_attempts=1).run(CapabilitySpec(raw_description='predict'), knowledge)
    assert trace['status'] == 'ok'
    assert advice['evidence_ids'] == ['Run_A']
    assert trace['evidence_id_corrections'] == {alias: 'Run_A'}
    _, rejected = AdvisorAgent(Model('Run_A:algorithm_logistic_regression__c1:v99'), max_attempts=1).run(CapabilitySpec(raw_description='predict'), knowledge)
    assert rejected['status'] == 'fallback' and 'hallucinated evidence' in rejected['attempts'][0]['error']
