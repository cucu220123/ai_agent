"""Summarize all preregistered trials; failed tasks remain in every rate denominator."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ablation.storage import atomic_json, sha, verify_guard

NAMES = {'full_system': 'Full', 'no_graph': 'w/o Graph', 'no_experience': 'w/o Experience',
         'no_beam': 'w/o Beam', 'single_candidate': 'Single Candidate',
         'no_repair': 'w/o Repair', 'no_knowledge': 'LLM-only'}


def stats(values):
    values = [float(value) for value in values if value is not None]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'std': statistics.stdev(values) if len(values) > 1 else None}


def aggregate(records):
    n = len(records)
    candidate_count = sum(row['candidate_count'] for row in records)
    repaired = sum(row['repair_triggered_candidates'] for row in records)
    metric_names = sorted({metric for row in records for metric in row['metrics']})
    measures = ['repair_rounds', 'candidate_count', 'candidate_execution_count', 'runtime',
                'llm_calls', 'llm_latency_seconds', 'repair_calls', 'codegen_retry_calls',
                'peak_candidate_memory_mb', 'planner_rank_of_winner', 'beam_states_expanded',
                'beam_states_pruned', 'graph_nodes', 'graph_edges', 'historical_cases',
                'failure_experiences', 'document_evidence', 'planner_cited_evidence',
                'retrieved_evidence_count']
    return {
        'total_runs': n, 'completed_runs': sum(row['completion'] for row in records),
        'completion_rate': sum(row['completion'] for row in records) / n,
        'first_pass_code_success_rate': sum(row['first_pass_code_success_count'] for row in records) / candidate_count if candidate_count else None,
        'first_pass_workflow_rate': sum(row['first_pass'] for row in records) / n,
        'candidate_success_coverage': sum(row['candidate_success_coverage'] for row in records) / n,
        'candidate_pass_rate': sum(row['candidate_pass_count'] for row in records) / candidate_count if candidate_count else None,
        'candidate_count_total': candidate_count,
        'repair_triggered_candidates': repaired,
        'repair_success_count': sum(row['repair_success_count'] for row in records),
        'repair_success_rate': sum(row['repair_success_count'] for row in records) / repaired if repaired else None,
        'rank_one_failed_other_passed_count': sum(row['rank_one_failed_other_passed'] for row in records),
        'winner_configurations': [row['winner_config'] for row in records if row['completion']],
        'metrics': {metric: stats([row['metrics'].get(metric) for row in records if row['completion']]) for metric in metric_names},
        'cost_and_evidence': {key: stats([row[key] for row in records]) for key in measures},
    }


def summarize(output: Path):
    protocol = json.loads((output / 'protocol.json').read_text())
    verify_guard(ROOT, protocol['guard'])
    records = []
    for trial in protocol['schedule']:
        path = output / 'trials' / trial['trial_id'] / 'result.json'
        if not path.exists():
            raise ValueError(f'incomplete schedule: {trial["trial_id"]}; do not report partial results as complete')
        record = json.loads(path.read_text())
        assert record['protocol_sha256'] == sha(output / 'protocol.json')
        assert not record['control_errors'] and record['frozen_guard_passed']
        for name, digest in record['source_manifest'].items():
            assert sha(path.parent / name) == digest, f'candidate source changed: {name}'
        records.append(record)
    groups = {dataset: {name: aggregate([row for row in records if row['dataset'] == dataset and row['experiment'] == name])
                       for name in NAMES} for dataset in protocol['datasets']}
    cross_task = {name: aggregate([row for row in records if row['experiment'] == name]) for name in NAMES}
    # Do not average AUC and accuracy across different tasks.
    for group in cross_task.values():
        del group['metrics']
    pairs = {}
    for dataset in protocol['datasets']:
        pairs[dataset] = {}
        full = {row['seed']: row for row in records if row['dataset'] == dataset and row['experiment'] == 'full_system'}
        metric = 'roc_auc' if dataset == 'customer_churn' else 'f1'
        for name in NAMES:
            if name == 'full_system':
                continue
            condition = {row['seed']: row for row in records if row['dataset'] == dataset and row['experiment'] == name}
            deltas = [full[seed]['metrics'][metric] - condition[seed]['metrics'][metric]
                      for seed in full if full[seed]['completion'] and condition[seed]['completion']]
            pairs[dataset][name] = {'quality_difference_full_minus_condition': stats(deltas),
                'paired_completed_seeds': [seed for seed in full if full[seed]['completion'] and condition[seed]['completion']],
                'completion_difference_full_minus_condition': groups[dataset]['full_system']['completion_rate'] - groups[dataset][name]['completion_rate']}
    return {'protocol_sha256': sha(output / 'protocol.json'), 'git_commit': protocol['execution_git_commit'],
            'setting_count': len(NAMES), 'dataset_count': len(groups), 'seed_count': len(protocol['seeds']),
            'total_trials': len(records), 'mode': protocol['mode'], 'groups': groups, 'cross_task': cross_task,
            'paired_comparisons': pairs, 'records': records,
            'quality_denominator': 'passing selected workflows only; n reported, missing failures never replaced with zero',
            'rate_denominator': 'all 3 preregistered trials per setting/dataset, including early failures',
            'first_pass_denominator': 'all started candidates; first Coder response accepted and first Validator PASS',
            'repair_denominator': 'candidates with at least one runtime Repair attempt; null means not triggered',
            'statistical_claim': 'descriptive mean and sample std, n=3 seeds; no significance claim',
            'frozen_final_test_used': False, 'frozen_guard_passed': True}


def fmt(stat, digits=3):
    if not stat or stat['mean'] is None:
        return '—'
    spread = f' ± {stat["std"]:.{digits}f}' if stat['std'] is not None else ''
    return f'{stat["mean"]:.{digits}f}{spread}'


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def table(summary, dataset):
    primary = 'ROC-AUC' if dataset == 'customer_churn' else 'Accuracy'
    key = 'roc_auc' if dataset == 'customer_churn' else 'accuracy'
    lines = [f'| Setting | {primary} | F1{" (weighted)" if dataset == "text" else ""} | Completion | First-pass code | Repair rounds | Candidates | Runtime (s) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, group in summary['groups'][dataset].items():
        cost = group['cost_and_evidence']
        quality = group['metrics']
        lines.append(f'| {NAMES[name]} | {fmt(quality.get(key))} | {fmt(quality.get("f1"))} | '
                     f'{group["completed_runs"]}/{group["total_runs"]} | {pct(group["first_pass_code_success_rate"])} | '
                     f'{fmt(cost["repair_rounds"],1)} | {fmt(cost["candidate_count"],1)} | {fmt(cost["runtime"],1)} |')
    return '\n'.join(lines)


def render_study(summary):
    lines = ['# 消融实验：开发集上的模块贡献分析', '',
        '本实验与 Final Acceptance、Independent Final Test 分开保存。只使用新合成客户数据和公开文本 development 数据，'
        '没有执行或反馈封存的 745 条最终测试。所有失败都进入完成率分母。', '',
        f'执行代码提交：`{summary["git_commit"]}`。7 settings × 2 datasets × 3 seeds = {summary["total_trials"]} 个预注册任务。', '',
        '## 协议与可复现范围', '',
        '全部设置的 Requirement、Planner、Coder、Explanation 以及触发的 Critic/Repair 使用真实本地 Qwen API；'
        '领域抽取与历史知识固定复用此前真实抽取/测量快照，没有重新调用 Extraction，也没有 frozen Planner/Coder、Mock 或模板获胜。', '',
        'seed=42/123/2026 控制每组共享的 75/25 分层开发划分以及模型请求 seed；temperature=0。'
        '可信稳定性 worker 保持相同的 estimator seeds [42,42,9]，所以表中的方差主要反映数据划分与生成路径变化，'
        '不是三个不同 estimator seeds 的独立训练试验。模型服务仍可能有非确定性。', '',
        '每个任务有独立 SQLite、GraphML、代码、LLM 原始输入输出和报告。各任务起点相同，不按运行顺序累积消融经验。'
        '经验 recency 使用固定参考时间；执行顺序以固定种子打乱，降低设置与时间顺序的混淆。', '',
        'A1 仅关闭图检索，文档与案例通道保留。A2 从全部检索通道去掉实测运行、失败、修复、实验来源及数值 prior，保留基础领域知识。'
        'A3 执行 Planner 原始 Top-K，不扩展状态；文本只注册一个算法，因此也减少执行预算。'
        'A4 仅执行原始 Planner rank-1 算法，仍在该算法配置中搜索。A5 禁止运行后的 Critic/Repair/再验证；'
        'Coder 内置最多两次静态契约检查修正是各组相同的生成预算，单独统计。A6 仅有当前需求和统一执行契约/算法白名单，'
        'current_user_requirement 引用不计为检索证据。', '',
        '## 指标定义与分母', '',
        '- Completion：最终选出的真实生成代码通过可信 Validator / 全部预注册任务。',
        '- First-pass code：第一次 Coder 响应通过静态门禁，且首次执行 PASS / 所有开始生成的候选；内部生成重试不算 first-pass。',
        '- Repair success：触发运行后修复且最终 PASS 的候选 / 触发修复的候选；未触发时为 null，不写成 100%。',
        '- Repair rounds：每个任务全部候选的实际 Repair 轮数总和。',
        '- Candidate success coverage：至少一个候选 PASS 的任务比例；另保存候选级通过率。',
        '- 质量均值仅对成功选出的算法计算并保留 n；早期失败为缺失值，不能用成功子集的高均值掩盖低完成率。',
        '- mean ± sample std；只有三个种子，不宣称统计显著性。配对质量差仅使用双方均完成的同种子任务。',
        '- Runtime 包含 Agent/检索/验证/写回，模型服务预加载不计入；内存是候选子进程峰值，不是整个 LLM 服务的 GPU 内存。', '',
        '## Customer Churn', '', table(summary, 'customer_churn'), '',
        '新合成数据 1200 条，900 条训练、300 条验证，阈值 ROC-AUC ≥ 0.80。不是企业业务效果评估。', '',
        '## Text Classification', '', table(summary, 'text'), '',
        '仅使用 development 2234 条：每种子训练 1675、验证 559；accuracy 与 weighted F1 均要求 ≥ 0.70。', '',
        '## 跨任务汇总', '', '| Setting | Completion | First-pass code | Repair success | LLM calls / task | Runtime / task (s) |',
        '|---|---:|---:|---:|---:|---:|']
    for name, group in summary['cross_task'].items():
        lines.append(f'| {NAMES[name]} | {group["completed_runs"]}/{group["total_runs"]} | {pct(group["first_pass_code_success_rate"])} | '
                     f'{pct(group["repair_success_rate"])} | {fmt(group["cost_and_evidence"]["llm_calls"],1)} | '
                     f'{fmt(group["cost_and_evidence"]["runtime"],1)} |')
    lines += ['', '## 模块对照观察', '']
    questions = [('no_graph','Q1. GraphRAG'), ('no_experience','Q2. Experience Memory'),
                 ('no_beam','Q3. Beam Search'), ('single_candidate','Q4. Multiple Candidate Execution'),
                 ('no_repair','Q5. Self-Repair'), ('no_knowledge','Q6. Full vs LLM-only')]
    for setting, title in questions:
        lines += [f'### {title}', '']
        for dataset in summary['groups']:
            full = summary['groups'][dataset]['full_system']
            other = summary['groups'][dataset][setting]
            pair = summary['paired_comparisons'][dataset][setting]
            lines.append(f'- {dataset}：Full 完成 {full["completed_runs"]}/3；{NAMES[setting]} 完成 {other["completed_runs"]}/3。'
                         f'配对质量差（Full − 对照，主指标）为 {fmt(pair["quality_difference_full_minus_condition"])}，'
                         f'n={pair["quality_difference_full_minus_condition"]["n"]}。')
        if setting == 'no_graph':
            lines.append('图节点/边与实际 Planner 引用数用于描述证据可追溯性，不能把引用数量直接当作决策正确率；预测指标差异按上表如实解释。')
        elif setting == 'no_experience':
            lines.append('历史案例、失败经验和 prior 分解保存在每次 observed.json 中；该设置也删除来源中的历史记录，避免替代通道泄漏。')
        elif setting == 'no_beam':
            lines.append('winner_config、expanded/pruned 状态保留在汇总 JSON。No Beam 的计算预算可能更少，不能将全部差异归因于搜索策略本身。')
        elif setting == 'single_candidate':
            cases = [row['trial_id'] for row in summary['records'] if row['experiment']=='full_system' and row['rank_one_failed_other_passed']]
            lines.append('Full 中 rank-1 算法全部失败而其他算法通过的任务：' + (', '.join(cases) if cases else '未观察到。') + '。')
        elif setting == 'no_repair':
            full = summary['cross_task']['full_system']
            lines.append(f'Full 共 {full["repair_triggered_candidates"]} 个候选触发修复，{full["repair_success_count"]} 个修复后通过。'
                         '未触发失败的运行不能证明 Repair 有效；本消融不注入故障，正式 controlled repair 与自然失败另行说明。')
        else:
            lines.append('LLM-only 保留执行契约、安全验证、相同搜索与修复预算。此对照考察检索上下文的综合作用，不是无约束的单次 prompt。')
        lines += ['']
    lines += ['## 原始结果与限制', '',
        '所有 result.json、observed.json、原始调用、代码版本和失败日志位于 `experiments/ablation/results/study_20260908/`；'
        '机器可读汇总与逐任务 CSV 位于 `docs/evidence/ablation_*`。指标未达标、接口错误和模型失败均不删除。', '',
        '每组只有三个种子和两个小型任务；LLM 输出、成功子集选择和不同执行预算会影响比较。'
        '本研究提供描述性证据，不证明普遍收益或统计显著提升。正式客户流失 AUC 0.9289 / F1 0.3934 与文本最终 Accuracy/F1 约 0.824 '
        '均来自另行封存的验收，不参与本表选择。', '']
    return '\n'.join(lines)


def write_outputs(summary, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / 'ablation_summary.json', summary)
    for dataset in summary['groups']:
        rows = [row for row in summary['records'] if row['dataset'] == dataset]
        keys = list(rows[0])
        with (destination / f'ablation_{dataset}.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value,(dict,list)) else value
                                 for key,value in row.items()})
    (ROOT / 'docs/ABLATION_STUDY.md').write_text(render_study(summary), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/ablation/results/study_20260908')
    parser.add_argument('--destination', type=Path, default=ROOT / 'docs/evidence')
    args = parser.parse_args()
    result = summarize(args.output.resolve())
    write_outputs(result, args.destination)
    print(json.dumps({'trials': result['total_trials'], 'cross_task': result['cross_task']}, ensure_ascii=False, indent=2))
