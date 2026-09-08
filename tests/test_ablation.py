"""Experiment controls must isolate knowledge and preserve failure denominators."""
import copy
import json
from pathlib import Path
import pytest
from app.agents.planner_agent import PlannerAgent
from app.knowledge.retriever import RetrieverAgent
from app.models import CapabilitySpec, KnowledgeContext
from app.search.beam import BeamSearchPlanner
from experiments.ablation.controls import (Controls, CatalogOnly, RequirementReference,
    DisabledGraph, SearchControl, DevelopmentValidator)
from experiments.ablation.storage import (without_experience, import_snapshot, sha,
    verify_guard, assert_experiment_output)
from experiments.ablation.measure import measure
from experiments.ablation.summarize_ablation import aggregate

ROOT = Path(__file__).parents[1]


@pytest.fixture
def prior():
    return json.loads((ROOT / 'experiments/ablation/fixtures/prior_snapshot.json').read_text())


@pytest.mark.parametrize('task', ['binary_classification', 'text_classification'])
def test_no_experience_removes_all_channels_but_keeps_domain_knowledge(prior, tmp_path, task):
    cleaned, removed = without_experience(prior)
    assert removed and cleaned['capabilities'] and cleaned['algorithms']
    assert not cleaned['validation_runs'] and not cleaned['experiences']
    store = import_snapshot(cleaned, tmp_path / 'private.sqlite')
    spec = CapabilitySpec(raw_description='customer churn text review', task_type=task)
    knowledge = RetrieverAgent(store).run(spec)
    context, trace = RequirementReference().build(spec, knowledge)
    assert not knowledge.historical_cases and not knowledge.experiences
    assert knowledge.graph_evidence['nodes'] and trace['final_evidence_ids']
    encoded = json.dumps(context)
    assert not any(row['id'] in encoded for row in prior['validation_runs'])
    assert 'historical_metrics' not in json.dumps(cleaned)


def test_graph_off_retains_independent_document_and_case_channels(prior, tmp_path):
    store = import_snapshot(prior, tmp_path / 'private.sqlite')
    retriever = RetrieverAgent(store)
    retriever.graph = DisabledGraph()
    result = retriever.run(CapabilitySpec(raw_description='predict customer churn'))
    assert result.historical_cases and result.semantic_evidence
    assert not result.graph_evidence['nodes'] and not result.graph_evidence['edges']


def test_llm_only_cites_actual_request_without_retrieved_knowledge():
    spec = CapabilitySpec(raw_description='a unique user requirement')
    knowledge = CatalogOnly().run(spec)
    context, trace = RequirementReference().build(spec, knowledge)
    assert trace['final_evidence_ids'] == ['current_user_requirement']
    assert context['current_request_reference']['raw_description'] == spec.raw_description
    assert knowledge.algorithms and not knowledge.semantic_evidence
    assert not context['graph_candidates'] and not context['similar_historical_runs']


def test_no_beam_executes_unmodified_planner_top_k():
    class ForbiddenSearch:
        def search(self, *args):
            raise AssertionError('Beam must not run')
    spec = CapabilitySpec(raw_description='churn')
    parents = PlannerAgent().run(spec, KnowledgeContext())
    selected, trace = SearchControl(ForbiddenSearch(), Controls(enable_beam_search=False)).search(parents, spec, 2)
    assert selected == parents[:2] and trace.expanded == 0 and not trace.pruned


def test_single_candidate_keeps_original_planner_rank_one_algorithm():
    spec = CapabilitySpec(raw_description='churn')
    parents = list(reversed(PlannerAgent().run(spec, KnowledgeContext())))
    search = SearchControl(BeamSearchPlanner(), Controls(enable_multi_candidate=False))
    selected, trace = search.search(parents, spec, 3)
    assert len(selected) == 1 and trace.expanded > 1
    assert selected[0].base_algorithm_id == parents[0].algorithm_id
    assert search.parents[0]['algorithm_id'] == parents[0].algorithm_id


def test_development_validator_uses_explicit_holdout_and_fixed_thresholds(tmp_path):
    from tests.test_final_holdout import selected_fixture
    from app.validation.runner import ValidationRunner
    report, train, holdout, source = selected_fixture(tmp_path)
    spec = CapabilitySpec(**report['spec'])
    validator = DevelopmentValidator(ValidationRunner(30), holdout, 123, {})
    result = validator.run(source, train, spec, 'TFIDF')
    assert result.status == 'passed', result.errors
    split = result.checks['evaluation_split']
    assert split['training_rows'] == 12
    assert result.checks['functional']['prediction_rows'] == 8
    assert split['mode'] == 'ablation_development_holdout' and split['split_seed'] == 123
    spec.metric_thresholds = {'accuracy': .1}
    with pytest.raises(ValueError, match='preregistered'):
        validator.run(source, train, spec, 'TFIDF')


def test_failed_tasks_stay_in_summary_and_untriggered_repair_is_null():
    failed = measure({}, [], [], [], {}, 10, 0)
    candidate = {'plan': {'algorithm_id':'b', 'config_variant':'default'},
        'validation': {'status':'passed','metrics':{'roc_auc':.9}}, 'code_source':'llm',
        'generation_trace': {'attempts':[{'status':'accepted'}]}, 'repair_history':[],
        'attempts':[{'validation':{'status':'passed'}}]}
    passed = measure({'selected_plan':{'algorithm_id':'b'}}, [candidate],
                     [{'algorithm_id':'a'},{'algorithm_id':'b'}], [], {}, 20, 1)
    group = aggregate([failed, failed, passed])
    assert group['completion_rate'] == pytest.approx(1/3)
    assert group['metrics']['roc_auc']['n'] == 1
    assert group['metrics']['roc_auc']['std'] is None
    assert group['repair_success_rate'] is None
    assert passed['planner_rank_of_winner'] == 2 and passed['rank_one_failed_other_passed']


def test_first_response_failure_cannot_be_counted_as_first_pass():
    candidate = {'plan':{'algorithm_id':'b','config_variant':'default'},
        'validation':{'status':'passed','metrics':{}}, 'code_source':'llm',
        'generation_trace':{'attempts':[{'status':'rejected'},{'status':'accepted'}]},
        'repair_history':[], 'attempts':[{'validation':{'status':'passed'}}]}
    result = measure({'selected_plan':{'algorithm_id':'b'}}, [candidate], [], [], {}, 1, 1)
    assert result['completion'] and not result['first_pass']
    assert result['first_validation_pass_count'] == 1


def test_private_store_import_is_non_destructive(prior, tmp_path):
    original = copy.deepcopy(prior)
    store = import_snapshot(prior, tmp_path / 'private.sqlite')
    assert prior == original and store.list_validation_runs(100)
    with pytest.raises(FileExistsError):
        import_snapshot(prior, store.db_path)


def test_no_repair_stops_after_first_failure_without_model_call(tmp_path):
    from app.agents.generator_agent import GeneratorAgent
    from app.agents.protocol import AgentRuntime
    from app.config import Settings
    from app.execution.candidates import CandidateExecutor
    from app.models import ValidationResult
    from experiments.ablation.controls import DisabledCritique
    class FailedValidator:
        calls = 0
        def run(self, *args):
            self.calls += 1
            return ValidationResult('failed', 'fixture', errors=['deliberate test-only failure'])
    class ForbiddenRepair:
        def repair(self, *args, **kwargs):
            raise AssertionError('Repair must not be called')
    settings = Settings(generated_dir=tmp_path/'generated', llm_provider='mock', max_repair_rounds=0)
    validator = FailedValidator()
    executor = CandidateExecutor(settings, GeneratorAgent(), validator, DisabledCritique(), ForbiddenRepair())
    spec = CapabilitySpec(raw_description='fixture')
    plan = PlannerAgent().run(spec, KnowledgeContext())[0]
    result = executor.run('fixture', plan, spec, KnowledgeContext(), tmp_path/'data.csv', AgentRuntime([]))
    assert result['validation']['status'] == 'failed'
    assert len(result['attempts']) == validator.calls == 1
    assert not result['repair_history']


def test_guard_rejects_mutation_and_output_escape(tmp_path):
    path = tmp_path / 'sealed.py'
    path.write_text('original')
    guard = {'sealed.py': sha(path)}
    verify_guard(tmp_path, guard)
    path.write_text('changed')
    with pytest.raises(RuntimeError, match='guard'):
        verify_guard(tmp_path, guard)
    with pytest.raises(ValueError):
        assert_experiment_output(tmp_path, tmp_path / 'examples/acceptance')
    assert_experiment_output(tmp_path, tmp_path / 'experiments/ablation/results/new-study')
