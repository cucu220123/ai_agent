import copy
import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest
from app.generation.templates import render_algorithm
from app.models import AlgorithmPlan, CapabilitySpec
from app.validation.dataset import profile_dataset
from app.validation.evaluation import evaluation_frames
from app.validation.holdout import FinalHoldoutEvaluator
from app.validation.runner import ValidationRunner


def selected_fixture(tmp_path):
    dev, final = tmp_path / 'dev.csv', tmp_path / 'final.csv'
    pd.DataFrame({'text': [f'{"wonderful good" if i % 2 else "awful bad"} dev{i}' for i in range(12)], 'label': [i % 2 for i in range(12)]}).to_csv(dev, index=False)
    pd.DataFrame({'text': [f'{"wonderful good" if i % 2 else "awful bad"} final{i}' for i in range(8)], 'label': [i % 2 for i in range(8)]}).to_csv(final, index=False)
    spec = CapabilitySpec(raw_description='test fixture', task_type='text_classification', target_column='label', feature_columns=['text'], metrics=['accuracy', 'f1'], metric_thresholds={}, output_schema={'prediction': 'int'}, output_columns=['prediction'], probability_output_required=False, dataset_profile=profile_dataset(dev, 'label'))
    plan = AlgorithmPlan('algorithm_tfidf_logistic_regression', 'TF-IDF Logistic Regression', 'fixture', ['tfidf'])
    source = tmp_path / 'candidate.py'
    source.write_text(render_algorithm(spec, plan))
    winner = {'plan': plan.to_dict(), 'algorithm_path': str(source), 'artifact_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    report = {'run_id': 'test_selection', 'spec': spec.to_dict(), 'validation': {'status': 'passed'}, 'selected_plan': plan.to_dict(), 'candidate_results': [winner]}
    return report, dev, final, source


def test_actual_frozen_evaluation_uses_all_training_rows_and_external_test(tmp_path):
    report, dev, final, source = selected_fixture(tmp_path)
    # Enforce full-data fitting inside the isolated generated module.
    code = source.read_text().replace('    if TEXT_COLUMN not in train_df.columns:', "    assert len(train_df) == 12, 'must fit all development rows'\n    if TEXT_COLUMN not in train_df.columns:")
    assert 'assert len(train_df) == 12' in code
    source.write_text(code)
    report['candidate_results'][0]['artifact_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    evaluator = FinalHoldoutEvaluator(ValidationRunner(30))
    result = evaluator.evaluate(report, dev, final, tmp_path / 'sealed')
    assert result['status'] == 'passed', result['validation']['errors']
    assert result['validation']['checks']['evaluation_split']['training_rows'] == 12
    assert result['validation']['checks']['functional']['prediction_rows'] == 8
    assert result['validation']['metrics']['accuracy'] == 1
    assert result['final_test_feedback_to_agents'] is False
    assert evaluator.evaluate(report, dev, final, tmp_path / 'sealed') == result
    changed = copy.deepcopy(report); changed['run_id'] = 'different_selection'
    with pytest.raises(ValueError, match='already committed'):
        evaluator.evaluate(changed, dev, final, tmp_path / 'sealed')


def test_duplicate_text_cannot_cross_final_split_even_with_new_label():
    dev = pd.DataFrame({'text': ['Good  Service'], 'label': [1]})
    final = pd.DataFrame({'text': [' good service '], 'label': [0]})
    with pytest.raises(ValueError, match='overlap'):
        evaluation_frames(dev, 'label', 'text_classification', final)


def test_final_guard_rejects_changed_source_and_interrupted_evaluation(tmp_path):
    report, dev, final, source = selected_fixture(tmp_path)
    evaluator = FinalHoldoutEvaluator(ValidationRunner(30))
    original = source.read_bytes(); source.write_bytes(original + b'\n# changed\n')
    with pytest.raises(ValueError, match='source changed'):
        evaluator.evaluate(report, dev, final, tmp_path / 'sealed')
    source.write_bytes(original)
    class Interrupted:
        def run(self, *args, **kwargs):
            raise RuntimeError('interrupted fixture')
    evaluator = FinalHoldoutEvaluator(Interrupted())
    with pytest.raises(RuntimeError, match='interrupted fixture'):
        evaluator.evaluate(report, dev, final, tmp_path / 'sealed')
    with pytest.raises(RuntimeError, match='automatic re-evaluation prohibited'):
        evaluator.evaluate(report, dev, final, tmp_path / 'sealed')
