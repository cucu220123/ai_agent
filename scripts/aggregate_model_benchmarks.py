from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "benchmark_1_5b.json", "benchmark_coder_3b.json", "benchmark_instruct_7b.json",
    "benchmark_qwen3_8b.json", "benchmark_instruct_14b.json", "benchmark_coder_30b.json",
]


def main() -> None:
    benchmarks = []
    scanned = {}
    for name in SOURCES:
        payload = json.loads((ROOT / "docs/evidence" / name).read_text(encoding="utf-8"))
        benchmarks.extend(payload.get("benchmarks", []))
        scanned.update({item["name"]: item for item in payload.get("scanned_models", [])})
    output = {
        "benchmark_protocol": ["Requirement JSON", "Planner JSON", "Python executable code", "traceback repair"],
        "selected_routing": {"requirement_extraction_planner": "Qwen2.5-14B-Instruct", "code_repair": "Qwen2.5-Coder-3B-Instruct", "lightweight_fallback": "Qwen2.5-1.5B-Instruct", "optional_exclusive_gpu": "Qwen3-Coder-30B-A3B-Instruct"},
        "benchmarks": benchmarks,
        "scanned_candidates": list(scanned.values()),
    }
    target = ROOT / "docs/evidence/local_model_benchmark.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Local Model Benchmark", "", "Strict success means schema/runtime gates passed; an LLM call alone is not success.", "", "| Model | Weight GB | Requirement | Planner | Executable code | Repair | Overall | Latency s | Peak GPU MB |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in benchmarks:
        meta = scanned.get(result["model"], {})
        tasks = result.get("tasks", {})
        cell = lambda task: "PASS" if tasks.get(task, {}).get("success") else "FAIL"
        lines.append(f"| {result['model']} | {meta.get('weight_gb', 0)} | {cell('requirement_json')} | {cell('planner_json')} | {cell('code_generation')} | {cell('traceback_repair')} | {result.get('success_rate', 0):.2f} | {result.get('latency_seconds', 0):.1f} | {result.get('peak_gpu_memory_mb', 0):.0f} |")
    lines.extend(["", "Selected routing: Qwen2.5-14B-Instruct for structured language tasks; Qwen2.5-Coder-3B-Instruct for code/repair. The 30B coder failed with CUDA OOM under shared-GPU conditions and remains optional for exclusive GPU deployment."])
    target.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()

