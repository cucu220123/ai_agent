from __future__ import annotations

import json
from pathlib import Path

from app.models import WorkflowResult


def write_report(result: WorkflowResult, reports_dir: str | Path) -> tuple[Path, Path]:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{result.run_id}.json"
    md_path = reports_dir / f"{result.run_id}.md"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    validation = result.validation
    lines = [
        f"# 算法能力工厂验证报告：{result.run_id}",
        "",
        f"- 能力：{result.spec.capability_name}",
        f"- 任务类型：{result.spec.task_type}",
        f"- 目标列：`{result.spec.target_column}`",
        f"- 选中方案：{result.selected_plan.algorithm_name if result.selected_plan else 'None'}",
        f"- 最终状态：**{validation.status if validation else 'unknown'}**",
        "",
        "## 需求解析",
        "",
        "```json",
        json.dumps(result.spec.to_dict(), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 候选方案",
        "",
    ]
    for plan in result.plans:
        lines.append(f"- **{plan.algorithm_name}**：{plan.rationale}；历史预期指标 `{plan.expected_metrics}`")
    if result.candidate_results:
        metric_columns = list(dict.fromkeys([*result.spec.metrics, *result.spec.metric_thresholds]))
        metric_columns = metric_columns[:4] or ["primary_metric"]
        lines.extend(["", "## 候选算法自动比较", "", "| 算法 | 代码来源 | 状态 | " + " | ".join(metric_columns) + " | 耗时(s) |", "|---|---|---|" + "---:|" * len(metric_columns) + "---:|"])
        for item in result.candidate_results:
            v = item["validation"]
            values = " | ".join(f"{v['metrics'].get(metric, 0.0):.4f}" for metric in metric_columns)
            lines.append(f"| {v['algorithm']} | {item.get('code_source', 'unknown')} | {v['status']} | {values} | {v['runtime_seconds']:.3f} |")
    if result.search_trace:
        lines.extend(["", "## 方案搜索", "", f"- 策略：`{result.search_trace.get('strategy')}`", f"- Beam width：`{result.search_trace.get('beam_width')}`", f"- 扩展候选数：`{result.search_trace.get('expanded')}`", f"- 入选：`{result.search_trace.get('selected')}`"])
    if validation:
        lines.extend(["", "## 验证结果", "", f"- 算法：{validation.algorithm}", f"- 耗时：{validation.runtime_seconds:.3f}s", f"- 指标：`{validation.metrics}`", "", "| 检查项 | 结果 | 说明 |", "|---|---:|---|"])
        for key, value in validation.checks.items():
            detail = value.get('message', value)
            lines.append(f"| {key} | {'通过' if value.get('passed') else '失败'} | {detail} |")
        if validation.errors:
            lines.extend(["", "### 错误/修复反馈", "", *[f"- {error}" for error in validation.errors]])
    if result.repair_history:
        lines.extend(["", "## 修复历史", "", *[f"- 第 {h.get('round')} 轮：{'; '.join(h.get('changes', []))}" for h in result.repair_history]])
    lines.extend(["", "## 知识沉淀", "", "本次验证结果、候选方案经验和源材料已写入 SQLite 知识库及 GraphML 图谱。", "", "## GraphRAG evidence", "", f"- Retrieval trace：`{result.knowledge.retrieval_trace}`", f"- Graph nodes/edges：`{len(result.knowledge.graph_evidence.get('nodes', []))}/{len(result.knowledge.graph_evidence.get('edges', []))}`", f"- Historical cases：`{len(result.knowledge.historical_cases)}`", "", "## LLM/搜索轨迹", "", f"- LLM：`{result.llm_trace}`", f"- 搜索：`{result.search_trace}`", "", "## Agent event log", "", *[f"- `{event.get('agent')}`: {event.get('status')}" for event in result.event_log]])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
