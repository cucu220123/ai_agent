from __future__ import annotations

from pathlib import Path

from app.generation.templates import render_algorithm
from app.models import AlgorithmPlan, CapabilitySpec


class GeneratorAgent:
    def run(self, run_dir: str | Path, spec: CapabilitySpec, plan: AlgorithmPlan) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "algorithm.py"
        path.write_text(render_algorithm(spec, plan), encoding="utf-8")
        (run_dir / "algorithm_meta.json").write_text(
            __import__("json").dumps({"spec": spec.to_dict(), "plan": plan.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

