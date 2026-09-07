"""Recover an explanation after a completed measured run; preserve its source report."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from app.models import CapabilitySpec, KnowledgeContext
from app.llm.security import sanitize


def recover_explanation(result: dict, workflow, output: Path) -> dict:
    if result['explanation']['status'] == 'ok':
        return result
    directory = output / 'report_revisions'
    directory.mkdir(exist_ok=True)
    original = directory / f"{result['run_id']}.before-explanation.json"
    if not original.exists():
        original.write_text(json.dumps(sanitize(result), ensure_ascii=False, indent=2), encoding='utf-8')
    call_start = len(workflow.llm.calls)
    winner = next(c for c in result['candidate_results'] if c['plan']['algorithm_id'] == result['selected_plan']['algorithm_id'])
    explanation = workflow.explainer.run(CapabilitySpec(**result['spec']), KnowledgeContext(**result['knowledge']), result['candidate_results'], winner)
    calls = [{**c, 'call_id': 'explanation_recovery/' + c['call_id']} for c in workflow.llm.calls[call_start:]]
    recovery = {'kind': 'explanation_only_recovery', 'timestamp': datetime.now(timezone.utc).isoformat(), 'previous_report': str(original), 'previous_report_sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'preserved_fields': ['run_id', 'spec', 'candidate_results', 'selected_plan', 'validation', 'writeback', 'generated_files'], 'new_explanation': explanation, 'llm_calls': calls}
    (directory / f"{result['run_id']}.explanation-recovery.json").write_text(json.dumps(sanitize(recovery),ensure_ascii=False,indent=2),encoding='utf-8')
    assert explanation['status'] == 'ok', explanation.get('error')
    result = {**result, 'explanation': explanation, 'report_revision': recovery}
    result['llm_trace']['calls'].extend(calls)
    Path(result['report_json']).write_text(json.dumps(sanitize(result),ensure_ascii=False,indent=2),encoding='utf-8')
    return result
