"""Explicit denominators and failure retention for workflow ablations."""
from __future__ import annotations
from typing import Any


def measure(report: dict, candidates: list[dict], parents: list[dict], calls: list[dict],
            search: dict, runtime: float, started_candidates: int) -> dict[str, Any]:
    successful = [c for c in candidates if c['validation']['status'] == 'passed'
                  and c['code_source'] in {'llm', 'repaired_llm'}]
    winner_id = (report.get('selected_plan') or {}).get('algorithm_id')
    winner = next((c for c in successful if c['plan']['algorithm_id'] == winner_id), None)
    completed = winner is not None
    first_pass = []
    first_validation_pass = []
    triggered = []
    repaired = []
    rounds = 0
    for candidate in candidates:
        attempt = candidate['attempts'][0]
        generated = candidate['generation_trace'].get('attempts', [])
        first_validation_pass.append(attempt['validation']['status'] == 'passed')
        first_pass.append(first_validation_pass[-1] and len(generated) == 1
                          and generated[0]['status'] == 'accepted')
        repair_history = candidate.get('repair_history', [])
        rounds += len(repair_history)
        if repair_history:
            triggered.append(candidate)
            if candidate in successful:
                repaired.append(candidate)
    knowledge = report.get('knowledge', {})
    graph = knowledge.get('graph_evidence', {})
    advice = knowledge.get('planner_advice', {})
    citations = [i for i in advice.get('evidence_ids', []) if i != 'current_user_requirement']
    parent_ids = [p.get('base_algorithm_id') or p['algorithm_id'] for p in parents]
    winner_base = (winner['plan'].get('base_algorithm_id') or winner['plan']['algorithm_id']) if winner else None
    parent_rank = parent_ids.index(winner_base) + 1 if winner_base in parent_ids else None
    attempts = [a for c in candidates for a in c['attempts']]
    repair_calls = [c for c in calls if c['purpose'] == 'repair']
    codegen_calls = [c for c in calls if c['purpose'] == 'code_generation']
    successful_parents = {(c['plan'].get('base_algorithm_id') or c['plan']['algorithm_id']) for c in successful}
    rank_one_failed_other_passed = bool(parent_ids and successful and parent_ids[0] not in successful_parents)
    count = started_candidates
    return {
        'completion': completed, 'metrics': winner['validation']['metrics'] if winner else {},
        'partial_candidate_metrics': [{'candidate': c['plan']['algorithm_id'],
                                      'status': c['validation']['status'], 'metrics': c['validation']['metrics']}
                                     for c in candidates],
        'first_pass': any(first_pass), 'first_pass_code_success_count': sum(first_pass),
        'first_pass_code_denominator': count, 'first_validation_pass_count': sum(first_validation_pass),
        'repair_triggered_candidates': len(triggered), 'repair_success_count': len(repaired),
        'repair_success_rate': len(repaired) / len(triggered) if triggered else None,
        'repair_rounds': rounds, 'repair_module_triggered': bool(triggered),
        'candidate_success_coverage': bool(successful), 'candidate_pass_count': len(successful),
        'candidate_count': count, 'candidate_execution_count': len(attempts),
        'planner_rank_of_winner': parent_rank, 'winner_id': winner_id if completed else None,
        'winner_config': winner['plan']['config_variant'] if winner else None,
        'rank_one_failed_other_passed': rank_one_failed_other_passed,
        'beam_states_expanded': search.get('expanded', 0),
        'beam_states_pruned': len(search.get('pruned', [])),
        'graph_nodes': len(graph.get('nodes', [])), 'graph_edges': len(graph.get('edges', [])),
        'historical_cases': len(knowledge.get('historical_cases', [])),
        'failure_experiences': len(knowledge.get('experiences', [])),
        'document_evidence': len(knowledge.get('semantic_evidence', [])),
        'planner_cited_evidence': len(citations), 'planner_cited_evidence_ids': citations,
        'retrieved_evidence_count': len(graph.get('nodes', [])) + len(knowledge.get('semantic_evidence', []))
                                    + len(knowledge.get('historical_cases', [])) + len(knowledge.get('experiences', [])),
        'runtime': runtime, 'llm_calls': len(calls),
        'llm_latency_seconds': sum(c.get('latency_ms', 0) for c in calls) / 1000,
        'llm_error_calls': sum(c.get('status') != 'ok' for c in calls),
        'repair_calls': len(repair_calls), 'codegen_calls': len(codegen_calls),
        'codegen_retry_calls': max(0, len(codegen_calls) - count),
        'peak_candidate_memory_mb': max((a['validation'].get('resource_usage', {}).get('max_rss_kb') or 0
                                         for a in attempts), default=0) / 1024,
        'explanation_status': report.get('explanation', {}).get('status'),
    }
