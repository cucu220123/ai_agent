"""Administrative stopping must not turn missing evidence into success or failure."""
import json

import pytest

from experiments.ablation.storage import sha
from experiments.ablation.summarize_ablation import (
    RUNTIME_DATABASES, aggregate, load_records, paired_comparison, verify_archive_guard,
)


def test_empty_group_has_no_completion_or_quality_estimate():
    group = aggregate([])
    assert group['total_runs'] == 0
    assert group['completion_rate'] is None
    assert group['metrics'] == {}
    assert group['repair_success_rate'] is None


def test_pairing_uses_only_shared_seeds_and_keeps_failed_pairs():
    full = {42: {'completion': True, 'metrics': {'f1': .9}},
            123: {'completion': True, 'metrics': {'f1': .8}}}
    baseline = {123: {'completion': False, 'metrics': {}},
                2026: {'completion': True, 'metrics': {'f1': .7}}}
    result = paired_comparison(full, baseline, 'f1')
    assert result['paired_terminal_seeds'] == [123]
    assert result['paired_completed_seeds'] == []
    assert result['quality_difference_full_minus_condition']['mean'] is None
    assert result['completion_difference_full_minus_condition'] == 1
    baseline[123] = {'completion': True, 'metrics': {'f1': .75}}
    result = paired_comparison(full, baseline, 'f1')
    assert result['quality_difference_full_minus_condition']['mean'] == pytest.approx(.05)
    assert result['quality_difference_full_minus_condition']['std'] is None


def test_partial_study_requires_complete_stop_ledger_and_keeps_failure(tmp_path):
    protocol = {'schedule': [{'trial_id': x} for x in ('finished', 'interrupted', 'unstarted')]}
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    digest = sha(tmp_path / 'protocol.json')
    folder = tmp_path / 'trials/finished'
    folder.mkdir(parents=True)
    record = {'trial_id': 'finished', 'completion': False, 'protocol_sha256': digest,
              'control_errors': [], 'frozen_guard_passed': True, 'source_manifest': {}}
    (folder / 'result.json').write_text(json.dumps(record))
    pending = tmp_path / 'trials/interrupted'
    pending.mkdir(parents=True)
    (pending / 'started.json').write_text('{}')
    with pytest.raises(ValueError, match='explicit stop'):
        load_records(tmp_path, protocol)
    decision = {'reason': 'administrative_time_budget_stop', 'protocol_sha256': digest,
                'trial_status': [{'trial_id': a, 'status': b} for a, b in
                                 [('finished', 'finished'), ('interrupted', 'interrupted'), ('unstarted', 'not_started')]]}
    (tmp_path / 'study_stop.json').write_text(json.dumps(decision))
    records, ledger = load_records(tmp_path, protocol)
    assert records == [record] and not records[0]['completion']
    assert [row['status'] for row in ledger] == ['finished', 'interrupted', 'not_started']
    (pending / 'result.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='state changed'):
        load_records(tmp_path, protocol)


def test_archive_guard_allows_only_missing_runtime_databases(tmp_path):
    source = tmp_path / 'sealed.py'
    source.write_text('original')
    database = sorted(RUNTIME_DATABASES)[0]
    guard = {'sealed.py': sha(source), database: 'expected-database-hash'}
    check = verify_archive_guard(tmp_path, guard)
    assert check['verified_files'] == 1
    assert check['unavailable_runtime_databases'] == [database]
    source.write_text('changed')
    with pytest.raises(RuntimeError, match='guard'):
        verify_archive_guard(tmp_path, guard)
    source.write_text('original')
    file = tmp_path / database
    file.parent.mkdir(parents=True)
    file.write_text('modified database')
    with pytest.raises(RuntimeError, match='guard'):
        verify_archive_guard(tmp_path, guard)


def test_archive_guard_never_skips_missing_committed_evidence(tmp_path):
    with pytest.raises(RuntimeError, match='guard'):
        verify_archive_guard(tmp_path, {'examples/acceptance_text_20260908/final_evaluation/result.json': 'hash'})
