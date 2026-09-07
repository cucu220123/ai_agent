"""Human report plus complete machine-readable evidence, always sanitized."""
from __future__ import annotations
import json
from pathlib import Path
from app.llm.security import sanitize
from app.models import WorkflowResult


def write_report(result: WorkflowResult, reports_dir: str | Path) -> tuple[Path, Path]:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = reports_dir / f"{result.run_id}.json", reports_dir / f"{result.run_id}.md"
    record = sanitize(result.to_dict())
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    v = record["validation"] or {}
    lines = [f"# 验证报告 {result.run_id}", "", f"任务：{result.spec.capability_name} / {result.spec.task_type}", "", f"状态：**{v.get('status')}**；严格真实模型模式：{record['llm_trace'].get('strict_real_mode')}", "", f"完整证据：[JSON]({json_path.name})", "", "## 当前候选实际比较", "", "| 搜索状态 | 代码来源 | 状态 | 实测指标 | 运行秒数 |", "|---|---|---|---|---:|"]
    for item in record["candidate_results"]:
        measured = item["validation"]
        lines.append(f"| {item['plan']['algorithm_id']} | {item['code_source']} | {measured['status']} | {measured['metrics']} | {measured['runtime_seconds']:.3f} |")
    for title, payload in [
        ("为什么选择该方案", record["explanation"]),
        ("结构化需求", record["spec"]),
        ("Beam Search：剪枝与探索", record["search_trace"]),
        ("功能、指标、稳定性、资源与鲁棒性", {"checks": v.get("checks"), "resource_usage": v.get("resource_usage"), "errors": v.get("errors")}),
        ("检索到的历史运行", [{"run_id": c.get("run_id"), "algorithm_id": c.get("algorithm_id"), "similarity": c.get("similarity"), "metrics": c.get("metrics")} for c in record["knowledge"]["historical_cases"]]),
        ("知识写回与版本", record["writeback"]),
        ("真实调用计量", record["llm_trace"].get("calls", [])),
    ]:
        lines.extend(["", f"## {title}", "", "```json", json.dumps(payload, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## 修复与不可变代码版本", ""])
    for candidate in record["candidate_results"]:
        for attempt in candidate["attempts"]:
            lines.append(f"- {attempt['version_id']}: {attempt['validation']['status']}, SHA256={attempt['code_hash']}")
        if candidate["repair_history"]:
            lines.extend(["", "```json", json.dumps(candidate["repair_history"], ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## 工作流轨迹", "", *[f"- [{event.get('agent')}] {event.get('status')}" for event in record["event_log"]], "", "限制：当前数据与切分上的实验结果不能保证生产效果；原型沙箱不是对抗恶意代码的生产安全边界。"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path

