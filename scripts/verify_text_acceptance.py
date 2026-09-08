"""Portable read-only proof checks for development selection and sealed final test."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_acceptance import assert_real
from app.validation.holdout import sha256


def verify(output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    report = json.loads((output / 'development.json').read_text())
    assert_real(report)
    policy = json.loads((output / 'evaluation_policy.json').read_text())
    final = json.loads((output / 'final_evaluation/result.json').read_text())
    seal = json.loads((output / 'final_evaluation/commitment.json').read_text())
    assert final['commitment'] == seal['commitment']
    commitment = final['commitment']
    assert commitment['selection_run_id'] == report['run_id']
    assert commitment['selection_report_sha256'] == hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    assert commitment['winner_id'] == report['selected_plan']['algorithm_id']
    assert sha256(output / 'final_evaluation/frozen_winner.py') == commitment['source_sha256']
    assert sha256(output / 'workspace/data/development.csv') == commitment['development_sha256'] == policy['development_sha256']
    assert sha256(root / 'data/uci_sentiment/final_test.csv') == commitment['final_test_sha256'] == policy['final_test_sha256']
    assert final['status'] == 'passed' and final['final_test_feedback_to_agents'] is False
    for metric, threshold in policy['metric_thresholds'].items():
        assert report['validation']['metrics'][metric] >= threshold
        assert final['validation']['metrics'][metric] >= threshold
        assert final['validation']['checks']['metrics']['thresholds'][metric] >= threshold
    assert final['validation']['checks']['evaluation_split']['mode'] == 'frozen_final'
    assert max(e['timestamp'] for e in report['event_log']) <= seal['timestamp']
    versions = 0
    for candidate in report['candidate_results']:
        for attempt in candidate['attempts']:
            relative = attempt['algorithm_path'].split('/workspace/', 1)[1]
            path = (output / 'workspace' / relative).resolve()
            assert path.is_relative_to((output / 'workspace').resolve())
            assert sha256(path) == attempt['code_hash']
            versions += 1
    winner = next(c for c in report['candidate_results'] if c['plan']['algorithm_id'] == commitment['winner_id'])
    assert winner['artifact_sha256'] == commitment['source_sha256']
    return {'status': 'passed', 'run_id': report['run_id'], 'source_versions': versions, 'llm_calls': len(report['llm_trace']['calls']), 'development_metrics': report['validation']['metrics'], 'final_metrics': final['validation']['metrics'], 'training_rows': final['validation']['checks']['evaluation_split']['training_rows'], 'final_rows': final['validation']['checks']['evaluation_split']['evaluation_rows'], 'final_feedback_to_agents': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('examples/acceptance_text_20260908'))
    args = parser.parse_args()
    try:
        result = verify(args.output)
    except Exception as exc:
        result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
