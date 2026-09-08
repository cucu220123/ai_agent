"""Run preregistered, isolated development-only trials through the existing workflow."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.ablation.controls import Controls, attach_controls, freeze_experience_clock
from experiments.ablation.storage import (atomic_json, assert_experiment_output, export_snapshot,
    git_commit, import_snapshot, protect_files, sha, verify_guard, without_experience)
from experiments.ablation.measure import measure

CONFIGS = ('full_system', 'no_graph', 'no_experience', 'no_beam',
           'single_candidate', 'no_repair', 'no_knowledge')
SEEDS = (42, 123, 2026)
DESCRIPTIONS = {
    'customer_churn': '根据客户年龄、地区、近30天登录次数、消费金额、投诉次数、会员等级和使用月数预测未来30天客户流失。'
        '这是表格二分类，目标字段 churn，必须输出 prediction 和 probability。'
        '比较 Logistic Regression、Random Forest、Gradient Boosting 三种候选。'
        '评价 ROC-AUC、F1、PR-AUC、Precision、Recall；唯一质量阈值 ROC-AUC >= 0.80。'
        '处理缺失值、类别不平衡和未见类别。不要自行增加其他质量阈值。',
    'text': '对客户评论进行情感二分类。领域 customer_reviews，任务 text_classification。'
        '输入 text，目标 label，输出 prediction；不要求概率。使用 TF-IDF + Logistic Regression，'
        '可以比较不同词表和正则化配置。评价 accuracy 和 weighted F1，两个指标均要求 >= 0.70。'
        '处理缺失文本、未见词和小批量。不要自行增加其他质量阈值。',
}
THRESHOLDS = {'customer_churn': {'roc_auc': .8}, 'text': {'accuracy': .7, 'f1': .7}}


def implementation_hashes():
    paths = [*ROOT.glob('app/**/*.py'), *ROOT.glob('experiments/ablation/*.py'),
             *ROOT.glob('experiments/ablation/configs/*.yaml')]
    return {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in sorted(paths)}


def prepare(output: Path, snapshot_path: Path):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from scripts.generate_demo_data import generate
    assert_experiment_output(ROOT, output)
    if (output / 'protocol.json').exists():
        return json.loads((output / 'protocol.json').read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError('new experiment output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    configs = {}
    for index, name in enumerate(CONFIGS):
        config = json.loads((ROOT / f'experiments/ablation/configs/{name}.yaml').read_text())
        assert config['id'] == f'A{index}' and config['name'] == name
        Controls(**config['controls'])
        configs[name] = config
    guard = protect_files(ROOT)
    inputs = output / 'inputs'
    inputs.mkdir()
    # A new synthetic population; no acceptance customer CSV is used for tuning.
    generate(inputs / 'customer_churn.csv', n_rows=1200, seed=20260909)
    text_source = ROOT / 'data/uci_sentiment/development.csv'
    (inputs / 'text.csv').write_bytes(text_source.read_bytes())
    assert len(pd.read_csv(inputs / 'text.csv')) == 2234
    (inputs / 'prior_snapshot.json').write_bytes(snapshot_path.read_bytes())
    snapshot = json.loads(snapshot_path.read_text())
    if any('683902e2a7cf' in row['id'] for row in snapshot['validation_runs']):
        raise ValueError('public text acceptance must not enter the prior snapshot')
    splits = {}
    for dataset, target in (('customer_churn', 'churn'), ('text', 'label')):
        frame = pd.read_csv(inputs / f'{dataset}.csv')
        for seed in SEEDS:
            train_ids, validation_ids = train_test_split(list(range(len(frame))), test_size=.25,
                random_state=seed, stratify=frame[target])
            split = inputs / f'{dataset}_{seed}'
            split.mkdir()
            frame.iloc[train_ids].to_csv(split / 'train.csv', index=False)
            frame.iloc[validation_ids].to_csv(split / 'validation.csv', index=False)
            item = {'dataset': dataset, 'seed': seed, 'target': target,
                    'source_sha256': sha(inputs / f'{dataset}.csv'),
                    'train_indices': train_ids, 'validation_indices': validation_ids,
                    'train_sha256': sha(split / 'train.csv'),
                    'validation_sha256': sha(split / 'validation.csv'),
                    'train_rows': len(train_ids), 'validation_rows': len(validation_ids)}
            atomic_json(split / 'split.json', item)
            splits[f'{dataset}_{seed}'] = item
    schedule = [{'experiment': name, 'dataset': dataset, 'seed': seed,
                 'trial_id': f'{configs[name]["id"]}_{dataset}_{seed}'}
                for name in CONFIGS for dataset in DESCRIPTIONS for seed in SEEDS]
    random.Random(20260908).shuffle(schedule)
    protocol = {'schema_version': 1, 'execution_git_commit': git_commit(ROOT),
        'registered_at': datetime.now(timezone.utc).isoformat(),
        'mode': 'real_llm_all_settings', 'seeds': list(SEEDS), 'settings': configs,
        'datasets': DESCRIPTIONS, 'thresholds': THRESHOLDS, 'schedule': schedule,
        'paired_split_policy': '75/25 stratified development split; same row indices across settings',
        'seed_scope': 'split seed and OpenAI-compatible request seed; temperature=0. '
                      'Existing trusted stability worker keeps training seeds [42,42,9] in every setting.',
        'experience_as_of': '2026-09-08T00:00:00+00:00',
        'knowledge_policy': 'same frozen previously measured knowledge snapshot, before public text acceptance; '
                            'no learning between ablation trials; extraction is frozen real output, not rerun',
        'beam_width': 3, 'max_repair_rounds': 3, 'validation_repeats': 3,
        'validation_timeout_seconds': 90, 'validation_memory_mb': 16384,
        'planning_context_max_chars': 12000, 'llm_timeout_seconds': 600,
        'instruction_model': 'Qwen2.5-14B-Instruct', 'coder_model': 'Qwen3-Coder-30B-A3B-Instruct',
        'embedding_model': 'shibing624-text2vec-base-chinese',
        'semantic_backend_policy': 'local embedding on CPU; explicit TF-IDF fallback recorded by Retriever',
        'codegen_gate_attempts': 2,
        'no_repair_definition': 'disable runtime Critic/Repair/revalidation; the common Coder static-gate '
                                'retry budget of 2 is unchanged and separately counted',
        'no_beam_definition': 'execute original Planner Top-K proposals without state expansion',
        'single_candidate_definition': 'restrict to original Planner rank-1 algorithm, retain Beam on its configurations',
        'failure_policy': 'retain every terminal result; no retries of failed trials; incomplete trials are explicit errors',
        'guard': guard, 'implementation_sha256': implementation_hashes(),
        'input_sha256': {str(path.relative_to(output)): sha(path) for path in sorted(inputs.rglob('*')) if path.is_file()},
        'splits': {key: {k:v for k,v in value.items() if not k.endswith('_indices')} for key,value in splits.items()},
        'environment': {package: importlib.metadata.version(package)
                        for package in ('pandas', 'numpy', 'scikit-learn', 'networkx', 'pydantic', 'openai')},
    }
    atomic_json(output / 'protocol.json', protocol)
    print(json.dumps({'prepared': str(output), 'trials': len(schedule),
                      'protocol_sha256': sha(output / 'protocol.json')}), flush=True)
    return protocol


def check_protocol(output: Path, protocol: dict):
    verify_guard(ROOT, protocol['guard'])
    for name, digest in protocol['implementation_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'implementation changed after preregistration: {name}')
    for name, digest in protocol['input_sha256'].items():
        if sha(output / name) != digest:
            raise RuntimeError(f'input changed after preregistration: {name}')


def execute_trial(output: Path, protocol: dict, item: dict):
    from app.config import Settings
    from app.workflow import AlgorithmFactoryWorkflow
    from app.llm.security import sanitize
    check_protocol(output, protocol)
    trial = output / 'trials' / item['trial_id']
    result_path = trial / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    trial.mkdir(parents=True, exist_ok=True)
    if (trial / 'started.json').exists():
        raise RuntimeError('interrupted trial retained; automatic rerun prohibited')
    config = protocol['settings'][item['experiment']]
    controls = Controls(**config['controls'])
    commitment = {**item, 'git_commit': protocol['execution_git_commit'],
                  'working_git_head': git_commit(ROOT), 'configuration': config,
                  'protocol_sha256': sha(output / 'protocol.json'),
                  'started_at': datetime.now(timezone.utc).isoformat()}
    atomic_json(trial / 'started.json', commitment)
    workspace = trial / 'workspace'
    workspace.mkdir()
    snapshot = json.loads((output / 'inputs/prior_snapshot.json').read_text())
    removed = []
    if not controls.enable_experience_retrieval or not controls.enable_knowledge_context:
        snapshot, removed = without_experience(snapshot)
    store = import_snapshot(snapshot, workspace / 'knowledge.sqlite')
    (trial / 'initial.graphml').write_bytes(store.graphml_path.read_bytes())
    atomic_json(trial / 'memory_control.json', {'removed_ids': removed,
        'source_snapshot_sha256': sha(output / 'inputs/prior_snapshot.json'),
        'retained_validation_runs': len(store.list_validation_runs(10000)),
        'retained_experiences': len(store.recent_experiences(10000))})
    split = output / 'inputs' / f'{item["dataset"]}_{item["seed"]}'
    data = workspace / 'data'
    data.mkdir()
    (data / 'train.csv').write_bytes((split / 'train.csv').read_bytes())
    # This is ablation validation data, never the sealed independent final test.
    validation = trial / 'development_validation.csv'
    validation.write_bytes((split / 'validation.csv').read_bytes())
    settings = Settings(project_root=workspace, data_dir=data,
        generated_dir=workspace / 'generated', reports_dir=workspace / 'reports',
        knowledge_db=workspace / 'knowledge.sqlite', graphml_path=workspace / 'knowledge.graphml',
        llm_provider='openai', secret_file=None, local_model_path=None,
        local_instruction_model_path=None, local_coder_model_path=None,
        openai_base_url=os.environ['OPENAI_BASE_URL'], openai_api_key='local-only',
        openai_model=protocol['instruction_model'], openai_coder_base_url=os.environ['OPENAI_CODER_BASE_URL'],
        openai_coder_model=protocol['coder_model'], strict_real_llm=True,
        codegen_mode='free_form_llm', validation_cv_folds=0, validation_seed_variance=.02,
        bootstrap_knowledge=False, llm_timeout_seconds=protocol['llm_timeout_seconds'],
        beam_width=protocol['beam_width'], max_repair_rounds=protocol['max_repair_rounds'] if controls.enable_repair else 0,
        planning_context_max_chars=protocol['planning_context_max_chars'],
        validation_timeout_seconds=protocol['validation_timeout_seconds'],
        validation_memory_mb=protocol['validation_memory_mb'], validation_repeats=protocol['validation_repeats'],
        embedding_model_path=os.environ.get('EMBEDDING_MODEL_PATH'))
    freeze_experience_clock(protocol['experience_as_of'])
    started = time.perf_counter()
    workflow, result, error = None, None, None
    with (trial / 'execution.log').open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                workflow = attach_controls(AlgorithmFactoryWorkflow(settings), controls, validation,
                    item['seed'], protocol['thresholds'][item['dataset']], trial / 'llm')
                result = workflow.run(protocol['datasets'][item['dataset']], data / 'train.csv',
                                      provider_note='preregistered development-only real-LLM ablation').to_dict()
            except Exception as exc:
                error = sanitize(f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
    elapsed = time.perf_counter() - started
    if result is None and settings.reports_dir.exists():
        reports = [path for path in settings.reports_dir.glob('*.json') if not path.name.endswith('.failure.json')]
        if reports:
            result = json.loads(reports[-1].read_text())
    report = result or {}
    if workflow:
        candidates = workflow.executor.results
        calls = workflow.llm.calls
        parents = workflow.searcher.parents
        search = workflow.searcher.last_trace or {}
        if workflow.retriever.last is not None:
            report.setdefault('knowledge', workflow.retriever.last.to_dict())
        count = len(workflow.executor.started)
    else:
        candidates, calls, parents, search, count = [], [], [], {}, 0
    # Preserve partial diagnostics even if the workflow stopped before final reporting.
    atomic_json(trial / 'observed.json', sanitize({'report': report, 'candidates': candidates,
                'initial_planner_plans': parents, 'search': search, 'llm_calls': calls, 'error': error}))
    measurements = measure(report, candidates, parents, calls, search, elapsed, count)
    knowledge = report.get('knowledge', {})
    control_errors = []
    if not controls.enable_graph_retrieval or not controls.enable_knowledge_context:
        if measurements['graph_nodes'] or measurements['graph_edges']:
            control_errors.append('graph retrieval was not disabled')
    if not controls.enable_experience_retrieval or not controls.enable_knowledge_context:
        if measurements['historical_cases'] or measurements['failure_experiences']:
            control_errors.append('experience retrieval was not disabled')
        forbidden = {row['id'] for row in json.loads((output / 'inputs/prior_snapshot.json').read_text())['validation_runs']}
        context = json.dumps(knowledge.get('planning_context', {}), ensure_ascii=False)
        if any(value in context for value in forbidden):
            control_errors.append('historical run leaked through another context channel')
    if not controls.enable_knowledge_context and (measurements['document_evidence'] or measurements['planner_cited_evidence']):
        control_errors.append('retrieved knowledge leaked into LLM-only condition')
    if not controls.enable_repair and any(c['purpose'] in {'repair', 'critique'} for c in calls):
        control_errors.append('runtime repair/critique LLM was not disabled')
    if not controls.enable_multi_candidate and count > 1:
        control_errors.append('single candidate budget exceeded')
    if not controls.enable_beam_search and measurements['beam_states_expanded']:
        control_errors.append('beam was not disabled')
    verify_guard(ROOT, protocol['guard'])
    value = {**commitment, **measurements, 'error': error, 'control_errors': control_errors,
             'model_mode': protocol['mode'], 'extraction_mode': 'frozen_previous_real_extraction',
             'completed_at': datetime.now(timezone.utc).isoformat(),
             'source_manifest': {str(path.relative_to(trial)): sha(path)
                                 for path in sorted((workspace / 'generated').rglob('*')) if path.is_file()},
             'frozen_guard_passed': True}
    atomic_json(result_path, sanitize(value))
    print(json.dumps({'trial': item['trial_id'], 'completion': value['completion'],
                      'metrics': value['metrics'], 'error': error, 'control_errors': control_errors}), flush=True)
    if control_errors:
        raise RuntimeError('ablation control violation; retain results and stop schedule')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/ablation/results/study_20260908')
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'experiments/ablation/fixtures/prior_snapshot.json')
    parser.add_argument('--export-prior', type=Path, help='one-time read-only SQLite export before preregistration')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-one', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    assert_experiment_output(ROOT, output)
    if args.export_prior:
        if args.snapshot.exists():
            raise FileExistsError('prior snapshot already frozen')
        export_snapshot(args.export_prior, args.snapshot)
        print('Exported read-only historical snapshot; no LLM call.')
        return
    protocol = prepare(output, args.snapshot)
    check_protocol(output, protocol)
    if args.prepare_only:
        return
    if args.run_one:
        item = next(item for item in protocol['schedule'] if item['trial_id'] == args.run_one)
        execute_trial(output, protocol, item)
        return
    # Each process gets fresh Python/embedding state; all share the same model service.
    os.environ.update(LLM_PROVIDER='openai', OPENAI_API_KEY='local-only',
                      OPENAI_MODEL=protocol['instruction_model'], OPENAI_CODER_MODEL=protocol['coder_model'],
                      AI_FACTORY_TRACE='1', EMBEDDING_DEVICE='cpu', TOKENIZERS_PARALLELISM='false',
                      OPENAI_CODER_API_KEY='local-only')
    # Refuse accidental cloud calls. Read-only service discovery does not run an acceptance.
    from urllib.parse import urlsplit
    from urllib.request import urlopen
    service_models = {}
    for key, expected in [('OPENAI_BASE_URL', protocol['instruction_model']),
                          ('OPENAI_CODER_BASE_URL', protocol['coder_model'])]:
        endpoint = os.environ[key].rstrip('/')
        if urlsplit(endpoint).hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('this protocol uses loopback model services only')
        with urlopen(endpoint + '/models', timeout=15) as response:
            models = [entry['id'] for entry in json.load(response)['data']]
        if expected not in models:
            raise ValueError(f'expected model unavailable: {expected}')
        service_models[key] = models
    if not (output / 'model_services.json').exists():
        atomic_json(output / 'model_services.json', {'models': service_models,
            'checked_at': datetime.now(timezone.utc).isoformat(), 'transport': 'loopback OpenAI-compatible API'})
    for index, item in enumerate(protocol['schedule']):
        trial = output / 'trials' / item['trial_id']
        if (trial / 'result.json').exists():
            existing = json.loads((trial / 'result.json').read_text())
            if existing.get('control_errors'):
                raise RuntimeError('cannot resume past an invalid control result')
            continue  # Failed completed trials are retained, never retried.
        print(f'[{index+1}/{len(protocol["schedule"])}] {item["trial_id"]}', flush=True)
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                  '--output', str(output), '--run-one', item['trial_id']])
        if process.returncode:
            raise RuntimeError(f'trial process failed ({process.returncode}); preserve all files before resume')
    verify_guard(ROOT, protocol['guard'])
    atomic_json(output / 'finished.json', {'status': 'complete', 'trials': len(protocol['schedule']),
                'timestamp': datetime.now(timezone.utc).isoformat(), 'frozen_guard_passed': True})


if __name__ == '__main__':
    main()
