"""Real text workflow on public data, followed by sealed final-test evaluation."""
from __future__ import annotations
import argparse
import json
import shutil
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings
from app.validation.holdout import FinalHoldoutEvaluator, sha256
from app.validation.runner import ValidationRunner
from app.workflow import AlgorithmFactoryWorkflow
from scripts.acceptance_recovery import recover_explanation
from scripts.run_acceptance import assert_real, save

ROOT = Path(__file__).resolve().parents[1]


def run(output: Path, provider: str, stage: str, prior_workspace: Path | None = None) -> dict:
    output = output.resolve()
    work = output / 'workspace'
    (work / 'data').mkdir(parents=True, exist_ok=True)
    data_root = ROOT / 'data/uci_sentiment'
    manifest = json.loads((data_root / 'manifest.json').read_text())
    for name in ('development', 'final_test'):
        assert sha256(data_root / f'{name}.csv') == manifest['splits'][name]['sha256']
    # Final rows stay outside the agent workspace and are never passed to run().
    development = work / 'data/development.csv'
    if not development.exists():
        shutil.copy2(data_root / 'development.csv', development)
    assert sha256(development) == manifest['splits']['development']['sha256']
    for name in ('business_material.md', 'text_material.md', 'reference_preprocessing.py'):
        shutil.copy2(ROOT / 'data' / name, work / 'data' / name)
    policy = {'dataset_archive_sha256': manifest['archive_sha256'], 'development_sha256': sha256(development), 'final_test_sha256': manifest['splits']['final_test']['sha256'], 'metric_thresholds': {'accuracy': .70, 'f1': .70}, 'beam_width': 3, 'max_repair_rounds': 3, 'selection': 'development validation only', 'final_test': 'evaluate frozen winner once; no repair or winner replacement from final metrics'}
    policy_path = output / 'evaluation_policy.json'
    if policy_path.exists():
        assert json.loads(policy_path.read_text()) == policy, 'evaluation policy changed'
    else:
        save(policy_path, policy)
    if prior_workspace and not (work / 'knowledge.sqlite').exists():
        with sqlite3.connect(prior_workspace.resolve().joinpath('knowledge.sqlite').as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(work / 'knowledge.sqlite') as destination:
            source.backup(destination)
        shutil.copy2(prior_workspace / 'knowledge.graphml', output / 'initial_knowledge.graphml')
        save(output / 'knowledge_initialization.json', {'source_workspace': str(prior_workspace.resolve()), 'graph_sha256': sha256(output / 'initial_knowledge.graphml'), 'policy': 'reuse previous measured runs and failures; no fabricated history'})
    selection_path = output / 'development.json'
    if stage in ('all', 'develop') and not selection_path.exists():
        settings = replace(get_settings(), project_root=work, data_dir=work / 'data', generated_dir=work / 'generated', reports_dir=work / 'reports', knowledge_db=work / 'knowledge.sqlite', graphml_path=work / 'knowledge.graphml', llm_provider=provider, strict_real_llm=True, beam_width=3, max_repair_rounds=3, planning_context_max_chars=18000)
        workflow = AlgorithmFactoryWorkflow(settings)
        description = ('Classify English product, film and restaurant review sentiment. This is text_classification. '
                       'The only feature is text; target label is 0/1. Use TF-IDF and Logistic Regression; compare preprocessing and regularization configurations. '
                       'Require weighted F1 >= 0.70 and accuracy >= 0.70 on development validation. '
                       'Output prediction only; probability is not required. Fill missing strings and handle unseen vocabulary, one-row and invalid inputs. '
                       'Report measured limitations; past small-data results are uncertain priors. A separate final test is unavailable during planning and repair.')
        result = workflow.run(description, development).to_dict()
        result = recover_explanation(result, workflow, output)
        assert_real(result)
        assert result['spec']['task_type'] == 'text_classification'
        assert all(result['spec']['metric_thresholds'].get(k, 0) >= v for k, v in policy['metric_thresholds'].items())
        save(selection_path, result)
        workflow.store.export_graph()
        shutil.copy2(settings.graphml_path, output / 'knowledge_after_development.graphml')
        save(output / 'extracted_knowledge.json', {'items': workflow.store.list_knowledge_items(1000)})
        print('[TextAcceptance] development PASS', flush=True)
    if stage in ('all', 'final'):
        result = json.loads(selection_path.read_text())
        assert_real(result)
        final = FinalHoldoutEvaluator(ValidationRunner(90)).evaluate(result, development, data_root / 'final_test.csv', output / 'final_evaluation')
        save(work / 'reports' / f"{result['run_id']}.final-evaluation.json", final)
        print('[TextAcceptance] final ' + final['status'].upper(), json.dumps(final['validation']['metrics']), flush=True)
        assert final['status'] == 'passed', final['validation']['errors']
    return {'output': str(output), 'stage': stage}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('examples/acceptance_text_20260908'))
    parser.add_argument('--provider', choices=['openai', 'auto', 'local'], default='auto')
    parser.add_argument('--stage', choices=['all', 'develop', 'final'], default='all')
    parser.add_argument('--prior-workspace', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.provider, args.stage, args.prior_workspace)))
