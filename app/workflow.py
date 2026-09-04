from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.agents.curator_agent import CuratorAgent
from app.agents.generator_agent import GeneratorAgent
from app.agents.parser_agent import ParserAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.repair_agent import RepairAgent
from app.config import Settings, get_settings
from app.knowledge.retriever import RetrieverAgent
from app.knowledge.store import KnowledgeStore
from app.llm.factory import build_llm
from app.models import WorkflowResult
from app.validation.report import write_report
from app.validation.runner import ValidationRunner


class AlgorithmFactoryWorkflow:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.store = KnowledgeStore(self.settings.knowledge_db, self.settings.graphml_path)
        seed = Path(__file__).parent / "knowledge" / "seed_data" / "knowledge.json"
        if not self.store.list_capabilities():
            self.store.seed_from_json(seed)
        self.parser = ParserAgent()
        self.retriever = RetrieverAgent(self.store)
        self.planner = PlannerAgent()
        self.generator = GeneratorAgent()
        self.repair = RepairAgent()
        self.validator = ValidationRunner(self.settings.validation_timeout_seconds)
        self.curator = CuratorAgent(self.store)
        self.llm = build_llm(self.settings)

    def run(self, description: str, data_path: str | Path, provider_note: str | None = None) -> WorkflowResult:
        run_id = uuid.uuid4().hex[:12]
        data_path = Path(data_path)
        spec = self.parser.run(description, str(data_path))
        knowledge = self.retriever.run(spec)
        plans = self.planner.run(spec, knowledge)
        if not plans:
            raise ValueError(f"no compatible algorithm plan for task type {spec.task_type}")
        # Call the configured provider for traceability/explanation. Code generation remains constrained by templates.
        try:
            llm_note = self.llm.complete("你是算法工厂规划助手。只返回简短说明。", json.dumps({"spec": spec.to_dict(), "knowledge": knowledge.to_dict()}, ensure_ascii=False))
        except Exception as exc:
            # A quota/network/model failure must not break the reproducible template path.
            llm_note = f"LLM provider unavailable; deterministic offline fallback used: {type(exc).__name__}: {exc}"
        candidate_results: list[dict] = []
        generated_files: list[str] = []
        for plan in plans:
            run_dir = self.settings.generated_dir / run_id / plan.algorithm_id
            generated_path = self.generator.run(run_dir, spec, plan)
            generated_files.extend([str(generated_path), str(generated_path.parent / "algorithm_meta.json")])
            history: list[dict] = []
            candidate_validation = None
            for round_no in range(self.settings.max_repair_rounds + 1):
                candidate_validation = self.validator.run(generated_path, data_path, spec, plan.algorithm_name, repair_round=round_no)
                if candidate_validation.status == "passed":
                    break
                if round_no >= self.settings.max_repair_rounds:
                    break
                feedback = "\n".join(candidate_validation.errors) + "\n" + candidate_validation.stderr
                history.append(self.repair.repair(generated_path, feedback, round_no + 1))
            candidate_validation.repair_round = len(history)
            candidate_validation.warnings.append(llm_note[:500])
            candidate_results.append({"plan": plan.to_dict(), "validation": candidate_validation.to_dict(), "repair_history": history, "algorithm_path": str(generated_path)})
        passed = [x for x in candidate_results if x["validation"]["status"] == "passed"]
        ranking = passed or candidate_results
        ranking.sort(key=lambda x: (x["validation"]["metrics"].get("roc_auc", -1.0), -x["validation"].get("runtime_seconds", 1e9)), reverse=True)
        winner = ranking[0]
        selected = next(p for p in plans if p.algorithm_id == winner["plan"]["algorithm_id"])
        # Preserve the exact candidate result, including warnings and repair details.
        validation = __import__("app.models", fromlist=["ValidationResult"]).ValidationResult(**winner["validation"])
        repair_history = winner["repair_history"]
        self.curator.run(run_id, spec, selected, validation, repair_history)
        result = WorkflowResult(run_id=run_id, spec=spec, knowledge=knowledge, plans=plans, selected_plan=selected, generated_files=generated_files, validation=validation, repair_history=repair_history, candidate_results=candidate_results)
        json_path, md_path = write_report(result, self.settings.reports_dir)
        result.report_json, result.report_markdown = str(json_path), str(md_path)
        # Rewrite JSON after adding report paths.
        json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result
