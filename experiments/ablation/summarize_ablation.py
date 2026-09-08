"""Audit completed or explicitly stopped studies without hiding missing trials."""
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

RUNTIME_DATABASES = {
    'examples/acceptance_real_20260907/workspace/knowledge.sqlite',
    'examples/acceptance_text_20260908/workspace/knowledge.sqlite',
}


def verify_archive_guard(root: Path, guard: dict) -> dict:
    """A Git clone omits runtime databases; present databases must still match."""
    absent = sorted(name for name in RUNTIME_DATABASES if name in guard and not (root / name).exists())
    available = {name: digest for name, digest in guard.items() if name not in absent}
    verify_guard(root, available)
    return {'verified_files': len(available), 'unavailable_runtime_databases': absent}


def load_records(output: Path, protocol: dict) -> tuple[list, list]:
    """Only an explicit, hash-bound administrative stop permits missing results."""
    digest = sha(output / 'protocol.json')
    stop_path = output / 'study_stop.json'
    stop = json.loads(stop_path.read_text(encoding='utf-8')) if stop_path.exists() else None
    if stop and (stop['protocol_sha256'] != digest or stop['reason'] != 'administrative_time_budget_stop'):
        raise ValueError('invalid administrative stop commitment')
    committed = {row['trial_id']: row['status'] for row in stop['trial_status']} if stop else {}
    if stop and set(committed) != {row['trial_id'] for row in protocol['schedule']}:
        raise ValueError('stop ledger must include every scheduled trial')
    records, ledger = [], []
    for trial in protocol['schedule']:
        folder = output / 'trials' / trial['trial_id']
        path = folder / 'result.json'
        status = 'finished' if path.exists() else ('interrupted' if (folder / 'started.json').exists() else 'not_started')
        if stop and committed[trial['trial_id']] != status:
            raise ValueError('trial state changed after administrative stop')
        if status != 'finished' and not stop:
            raise ValueError(f'incomplete schedule: {trial["trial_id"]}; explicit stop ledger required')
        ledger.append({**trial, 'status': status})
        if status != 'finished':
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['protocol_sha256'] == digest
        assert not record['control_errors'] and record['frozen_guard_passed']
        for name, expected in record['source_manifest'].items():
            assert sha(folder / name) == expected, f'candidate source changed: {name}'
        records.append(record)
    return records, ledger


def paired_comparison(full: dict, condition: dict, metric: str) -> dict:
    shared = sorted(full.keys() & condition.keys())
    valid = [seed for seed in shared if full[seed]['completion'] and condition[seed]['completion']]
    return {
        'quality_difference_full_minus_condition': stats([
            full[seed]['metrics'][metric] - condition[seed]['metrics'][metric] for seed in valid]),
        'paired_completed_seeds': valid,
        'paired_terminal_seeds': shared,
        'completion_difference_full_minus_condition': statistics.mean([
            int(full[seed]['completion']) - int(condition[seed]['completion']) for seed in shared]) if shared else None,
    }

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
        'completion_rate': sum(row['completion'] for row in records) / n if n else None,
        'first_pass_code_success_rate': sum(row['first_pass_code_success_count'] for row in records) / candidate_count if candidate_count else None,
        'first_pass_workflow_rate': sum(row['first_pass'] for row in records) / n if n else None,
        'candidate_success_coverage': sum(row['candidate_success_coverage'] for row in records) / n if n else None,
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
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    guard_check = verify_archive_guard(ROOT, protocol['guard'])
    records, ledger = load_records(output, protocol)
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
            pairs[dataset][name] = paired_comparison(full, condition, metric)
    return {'protocol_sha256': sha(output / 'protocol.json'), 'git_commit': protocol['execution_git_commit'],
            'setting_count': len(NAMES), 'dataset_count': len(groups), 'seed_count': len(protocol['seeds']),
            'total_trials': len(records), 'planned_trials': len(ledger),
            'interrupted_trials': sum(row['status'] == 'interrupted' for row in ledger),
            'not_started_trials': sum(row['status'] == 'not_started' for row in ledger),
            'trial_status': ledger, 'archive_guard_check': guard_check,
            'analysis_script_sha256': sha(Path(__file__)),
            'mode': protocol['mode'], 'groups': groups, 'cross_task': cross_task,
            'paired_comparisons': pairs, 'records': records,
            'quality_denominator': 'passing selected workflows only; n reported, missing failures never replaced with zero',
            'rate_denominator': 'all terminal observed trials, including failures; administrative interruptions and unstarted trials listed separately, not algorithm failures',
            'first_pass_denominator': 'all started candidates; first Coder response accepted and first Validator PASS',
            'repair_denominator': 'candidates with at least one runtime Repair attempt; null means not triggered',
            'statistical_claim': 'descriptive mean and sample std at actual observed n; incomplete randomized schedule, no significance claim',
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
    lines = [f'| Setting | n / planned | {primary} | F1{" (weighted)" if dataset == "text" else ""} | Completion | First-pass code | Repair rounds | Candidates | Runtime (s) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, group in summary['groups'][dataset].items():
        cost = group['cost_and_evidence']
        quality = group['metrics']
        lines.append(f'| {NAMES[name]} | {group["total_runs"]}/{summary["seed_count"]} | {fmt(quality.get(key))} | {fmt(quality.get("f1"))} | '
                     f'{group["completed_runs"]}/{group["total_runs"]} | {pct(group["first_pass_code_success_rate"])} | '
                     f'{fmt(cost["repair_rounds"],1)} | {fmt(cost["candidate_count"],1)} | {fmt(cost["runtime"],1)} |')
    return '\n'.join(lines)



def coverage_table(summary):
    lines = ['| Setting | Churn completed seeds | Text completed seeds | Interrupted |',
             '|---|---|---|---|']
    for name in NAMES:
        seeds = {d: sorted(row['seed'] for row in summary['records'] if row['experiment'] == name and row['dataset'] == d)
                 for d in summary['groups']}
        interrupted = [row['trial_id'] for row in summary['trial_status'] if row['status'] == 'interrupted'
                       and row['experiment'] == name]
        lines.append(f"| {NAMES[name]} | {seeds['customer_churn']} | {seeds['text']} | {', '.join(interrupted) or '—'} |")
    return lines


def operational_table(summary):
    lines = ['| Setting | Graph nodes / edges | Historical cases | Planner citations | Expanded / pruned | Candidates PASS | Repairs PASS / triggered |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, g in summary['cross_task'].items():
        c = g['cost_and_evidence']
        lines.append(f"| {NAMES[name]} | {fmt(c['graph_nodes'],1)} / {fmt(c['graph_edges'],1)} | "
                     f"{fmt(c['historical_cases'],1)} | {fmt(c['planner_cited_evidence'],1)} | "
                     f"{fmt(c['beam_states_expanded'],1)} / {fmt(c['beam_states_pruned'],1)} | "
                     f"{pct(g['candidate_pass_rate'])} | {g['repair_success_count']}/{g['repair_triggered_candidates']} |")
    examples = [(row['trial_id'], row['repair_success_count'], row['repair_triggered_candidates'])
                for row in summary['records'] if row['repair_triggered_candidates']]
    lines += ['', '自然修复触发任务（成功候选/触发候选）：' + '; '.join(f'{name}: {success}/{triggered}' for name, success, triggered in examples) + '。',
              '', '上述检索/搜索统计跨两个任务汇总，受完成种子的组成差异影响，只描述实际执行规模，不据此推断质量因果提升。'
              '完整工作流有替代候选时，修复失败不一定导致工作流失败；因此同时报告候选失败和任务完成，避免隐藏失败成本。']
    return lines


def render_study(summary):
    lines = ['# 消融实验：开发集上的模块贡献分析', '',
        '本实验与 Final Acceptance、Independent Final Test 分开保存。只使用新合成客户数据和公开文本 development 数据，'
        '没有执行或反馈封存的 745 条最终测试。所有已结束的失败任务都进入观测完成率分母；行政中断与未启动任务单独列出。', '',
        f'执行代码提交：`{summary["git_commit"]}`。预注册 7 settings × 2 datasets × 3 seeds = {summary["planned_trials"]} 个任务；实际结束 {summary["total_trials"]}，中断 {summary["interrupted_trials"]}，未启动 {summary["not_started_trials"]}。', '',
        '实验因执行时间预算提前结束，停止位置来自原先随机排程，没有根据分数挑选保留结果。原始 42 次协议保持不变。'
        '两个数据集均未完成全部 21 次，因此以下为不完整探索性消融，不能当成完整三种子验证。'
        '停止决策与逐任务状态见 study_stop.json。', '',
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
        'LLM calls 统计应用层调用次数，传输层 retry count 保存在逐次 trace；中断任务成本保留原始记录，未混入完整任务平均值。', '',
        '## 指标定义与分母', '',
        '- Completion：最终选出的真实生成代码通过可信 Validator / 已结束的观测任务（包含算法失败）。未运行和行政中断不伪装成算法失败，另给总体任务进度。',
        '- First-pass code：第一次 Coder 响应通过静态门禁，且首次执行 PASS / 所有开始生成的候选；内部生成重试不算 first-pass。',
        '- Repair success：触发运行后修复且最终 PASS 的候选 / 触发修复的候选；未触发时为 null，不写成 100%。',
        '- Repair rounds：每个任务全部候选的实际 Repair 轮数总和。',
        '- Candidate success coverage：至少一个候选 PASS 的任务比例；另保存候选级通过率。',
        '- 质量均值仅对成功选出的算法计算并保留 n；早期失败为缺失值，不能用成功子集的高均值掩盖低完成率。',
        '- mean ± sample std；各组实际 n=0–3，n=1 不计算 std，不宣称统计显著性。配对质量差仅使用双方均完成的同种子任务。',
        '- Runtime 包含 Agent/检索/验证/写回，模型服务预加载不计入；内存是候选子进程峰值，不是整个 LLM 服务的 GPU 内存。', '',
        '## 已执行覆盖范围', '', *coverage_table(summary), '',
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
    lines += ['', '## 检索、搜索与失败诊断', '', *operational_table(summary), '',
              '## 配对观察的解释', '',
              '不同设置的已结束种子不完全相同，组均值不能直接进行因果比较。下面仅比较同数据集、同种子、双方已结束的任务。'
              '零差表示该配对没有观察到预测收益，负差表示 Full 在该配对更低；不将证据引用数或历史案例数转换为质量提升结论。', '',
              '所有已结束任务都有通过的 winner，因此这些样本没有区分各设置的工作流完成率。'
              '候选级失败、修复失败和开销仍保留，不能把工作流完成率解释为每份生成代码都正确。', '',
              '## 模块对照观察', '']
    questions = [('no_graph','Q1. GraphRAG'), ('no_experience','Q2. Experience Memory'),
                 ('no_beam','Q3. Beam Search'), ('single_candidate','Q4. Multiple Candidate Execution'),
                 ('no_repair','Q5. Self-Repair'), ('no_knowledge','Q6. Full vs LLM-only')]
    for setting, title in questions:
        lines += [f'### {title}', '']
        for dataset in summary['groups']:
            full = summary['groups'][dataset]['full_system']
            other = summary['groups'][dataset][setting]
            pair = summary['paired_comparisons'][dataset][setting]
            lines.append(f'- {dataset}：Full 观测完成 {full["completed_runs"]}/{full["total_runs"]}；{NAMES[setting]} 观测完成 {other["completed_runs"]}/{other["total_runs"]}。'
                         f'配对质量差（Full − 对照，主指标）为 {fmt(pair["quality_difference_full_minus_condition"])}，'
                         f'n={pair["quality_difference_full_minus_condition"]["n"]}。')
        if setting == 'no_graph':
            lines.append('图节点/边与实际 Planner 引用数用于描述证据可追溯性，不能把引用数量直接当作决策正确率；预测指标差异按上表如实解释。')
        elif setting == 'no_experience':
            lines.append('历史案例、失败经验和 prior 分解保存在每次 observed.json 中；该设置也删除来源中的历史记录，避免替代通道泄漏。')
        elif setting == 'no_beam':
            lines.append('winner_config、expanded/pruned 状态保留在汇总 JSON。winner_config 是方案标签，free-form LLM 可能偏离方案，不能将它当作真实代码参数一致性的证明。No Beam 的计算预算可能更少，不能将全部差异归因于搜索策略本身。')
        elif setting == 'single_candidate':
            cases = [row['trial_id'] for row in summary['records'] if row['experiment']=='full_system' and row['rank_one_failed_other_passed']]
            lines.append('Full 中 rank-1 算法全部失败而其他算法通过的任务：' + (', '.join(cases) if cases else '未观察到。') + '。')
        elif setting == 'no_repair':
            full = summary['cross_task']['full_system']
            lines.append(f'Full 共 {full["repair_triggered_candidates"]} 个候选触发修复，{full["repair_success_count"]} 个修复后通过。'
                         '未触发失败的运行不能证明 Repair 有效；本消融不注入故障，正式 controlled repair 与自然失败另行说明。'
                         '比较 with/without repair 的任务完成率并没有在当前已结束样本中显示优势；恢复成功案例也应和失败的修复一起解释。')
        else:
            lines.append('LLM-only 保留执行契约、安全验证、相同搜索与修复预算。此对照考察检索上下文的综合作用，不是无约束的单次 prompt。')
        lines += ['']
    lines += ['## 原始结果与限制', '',
        '所有 result.json、observed.json、原始调用、代码版本和失败日志位于 `experiments/ablation/results/study_20260908/`；'
        '机器可读汇总与逐任务 CSV 位于 `docs/evidence/ablation_*`。指标未达标、接口错误和模型失败均不删除。'
        '中断任务保留已有调用与工作文件，未产生最终结果，故不编造质量分数。汇总脚本在停止后增加不完整研究的分母与状态处理，执行时实现哈希仍以原 protocol 为准。'
        'Git clone 通常不含两份未跟踪的正式运行 SQLite；离线汇总仅允许这两条确切路径缺失，并明确列为未验证。其他源码/数据缺失或哈希变化均拒绝。原运行服务器上两份数据库均通过原始严格 guard。', '',
        '原计划每组三个种子，提前结束后多数设置只有一到两个观测，且仅有两个小型任务；LLM 输出、成功子集选择和不同执行预算会影响比较。'
        '本研究提供描述性证据，不证明普遍收益或统计显著提升。正式客户流失 AUC 0.9289 / F1 0.3934 与文本最终 Accuracy/F1 约 0.824 '
        '均来自另行封存的验收，不参与本表选择。', '']
    return '\n'.join(lines)


def write_outputs(summary, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / 'ablation_summary.json', summary)
    for dataset in summary['groups']:
        rows = [row for row in summary['records'] if row['dataset'] == dataset]
        keys = list(rows[0]) if rows else ['trial_id', 'dataset', 'seed', 'completion']
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
