from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.agents.curator_agent import CuratorAgent
from app.agents.critic_agent import CriticAgent
from app.agents.advisor_agent import AdvisorAgent
from app.agents.generator_agent import GeneratorAgent
from app.agents.parser_agent import ParserAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.repair_agent import RepairAgent
from app.agents.requirement_agent import RequirementUnderstandingAgent
from app.config import Settings, get_settings
from app.knowledge.retriever import RetrieverAgent
from app.knowledge.store import KnowledgeStore
from app.llm.factory import build_llm
from app.llm.contracts import complete_with_trace
from app.plugins.registry import DEFAULT_REGISTRY
from app.search.beam import BeamSearchPlanner
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
        self.store.ensure_catalog_nodes()
        self.parser = ParserAgent()
        self.retriever = RetrieverAgent(self.store)
        self.planner = PlannerAgent()
        self.llm = build_llm(self.settings)
        self.generator = GeneratorAgent(self.llm, self.settings.llm_provider)
        self.repair = RepairAgent(self.llm, self.settings.llm_provider)
        self.validator = ValidationRunner(self.settings.validation_timeout_seconds)
        self.curator = CuratorAgent(self.store)
        self.critic = CriticAgent(self.llm, self.settings.llm_provider)
        self.advisor = AdvisorAgent(self.llm)
        self.requirement_agent = RequirementUnderstandingAgent(self.llm, self.settings.llm_provider)
        self.searcher = BeamSearchPlanner()

    def run(self, description: str, data_path: str | Path, provider_note: str | None = None) -> WorkflowResult:
        run_id = uuid.uuid4().hex[:12]
        data_path = Path(data_path)
        event_log: list[dict] = []
        event_log.append({"agent": "RequirementAgent", "status": "started"})
        spec, requirement_trace = self.requirement_agent.run(description, data_path)
        event_log.append({"agent": "RequirementAgent", "status": requirement_trace.get("status"), "trace": requirement_trace})
        knowledge = self.retriever.run(spec)
        event_log.append({"agent": "RetrievalAgent", "status": "ok", "trace": knowledge.retrieval_trace})
        advisor_trace = None
        advice_response = ""
        try:
            advice_response, advisor_trace = complete_with_trace(
                self.llm, self.settings.llm_provider, getattr(self.llm, "model", None), "structured_plan_advice",
                "你是严格的算法方案顾问。只输出 JSON 对象。", json.dumps({"spec": spec.to_dict(), "knowledge": knowledge.to_dict()}, ensure_ascii=False)
            )
            # Parse again through the shared contract helper; invalid advice is ignored.
            from app.llm.contracts import extract_json_object
            advice = extract_json_object(advice_response) or {}
            allowed = {p.id for p in DEFAULT_REGISTRY.algorithms_for(spec.task_type)}
            advised = [x for x in advice.get("candidate_algorithms", []) if x in allowed]
            if advised:
                spec.candidate_algorithms = advised
            if isinstance(advice.get("metric_thresholds"), dict):
                spec.metric_thresholds.update({str(k): float(v) for k, v in advice["metric_thresholds"].items() if isinstance(v, (int, float)) and 0 <= float(v) <= 1})
        except Exception:
            advice_response, advisor_trace = "", None
        plans = self.planner.run(spec, knowledge)
        if not plans:
            raise ValueError(f"no compatible algorithm plan for task type {spec.task_type}")
        beam_width = min(len(plans), max(1, int(__import__("os").getenv("BEAM_WIDTH", "3"))))
        plans, search_trace = self.searcher.search(plans, spec, beam_width)
        event_log.append({"agent": "PlannerAgent", "status": "ok", "plans": [p.algorithm_id for p in plans], "search": search_trace.to_dict()})
        # Code generation remains constrained by templates; the structured response is retained only as advice/trace.
        llm_trace = advisor_trace
        llm_note = advice_response or "LLM 未返回可用说明，已使用确定性模板。"
        if llm_trace and llm_trace.status == "fallback":
            llm_note = f"LLM 不可用，已使用确定性离线模板：{llm_trace.error}"
        candidate_results: list[dict] = []
        generated_files: list[str] = []
        for plan_index, plan in enumerate(plans):
            run_dir = self.settings.generated_dir / run_id / plan.algorithm_id
            generated_path = self.generator.run(run_dir, spec, plan, allow_llm=plan_index == 0)
            event_log.append({"agent": "CoderAgent", "status": "generated", "algorithm": plan.algorithm_id, "path": str(generated_path)})
            generated_files.extend([str(generated_path), str(generated_path.parent / "algorithm_meta.json")])
            history: list[dict] = []
            candidate_validation = None
            for round_no in range(self.settings.max_repair_rounds + 1):
                candidate_validation = self.validator.run(generated_path, data_path, spec, plan.algorithm_name, repair_round=round_no)
                event_log.append({"agent": "ValidatorAgent", "status": candidate_validation.status, "algorithm": plan.algorithm_id, "round": round_no, "errors": candidate_validation.errors[:3]})
                if candidate_validation.status == "passed":
                    break
                if round_no >= self.settings.max_repair_rounds:
                    break
                diagnosis = self.critic.run(spec, plan, candidate_validation, generated_path.read_text(encoding="utf-8"))
                event_log.append({"agent": "CriticAgent", "status": "ok", "algorithm": plan.algorithm_id, "diagnosis": diagnosis})
                feedback = "\n".join(candidate_validation.errors) + "\n" + candidate_validation.stderr + "\n诊断：" + json.dumps(diagnosis, ensure_ascii=False)
                repair_event = self.repair.repair(generated_path, feedback, round_no + 1)
                repair_event["diagnosis"] = diagnosis
                history.append(repair_event)
                event_log.append({"agent": "RepairAgent", "status": "applied", "algorithm": plan.algorithm_id, "round": round_no + 1})
            candidate_validation.repair_round = len(history)
            if candidate_validation.status != "passed" and plan_index == 0:
                recovery_path = self.generator.run(run_dir / "template_recovery", spec, plan, allow_llm=False)
                recovery_validation = self.validator.run(recovery_path, data_path, spec, plan.algorithm_name, repair_round=len(history) + 1)
                event_log.append({"agent": "RecoveryTemplate", "status": recovery_validation.status, "algorithm": plan.algorithm_id, "reason": "LLM proposal/repair did not pass; explicit deterministic recovery"})
                history.append({"round": len(history) + 1, "changes": ["explicit deterministic template recovery after LLM failure"], "provider": "template_recovery"})
                if recovery_validation.status == "passed":
                    generated_path = recovery_path
                    candidate_validation = recovery_validation
            candidate_validation.resource_usage["code_hash"] = self.store.artifact_fingerprint(generated_path)
            candidate_validation.warnings.append(llm_note[:500])
            candidate_results.append({"plan": plan.to_dict(), "validation": candidate_validation.to_dict(), "repair_history": history, "algorithm_path": str(generated_path), "artifact_sha256_16": self.store.artifact_fingerprint(generated_path)})
        passed = [x for x in candidate_results if x["validation"]["status"] == "passed"]
        ranking = passed or candidate_results
        primary_metric = "roc_auc" if spec.task_type == "binary_classification" else ("r2" if spec.task_type == "regression" else "anomaly_rate")
        ranking.sort(key=lambda x: (x["validation"]["metrics"].get(primary_metric, -1.0), -x["validation"].get("runtime_seconds", 1e9)), reverse=True)
        winner = ranking[0]
        selected = next(p for p in plans if p.algorithm_id == winner["plan"]["algorithm_id"])
        # Preserve the exact candidate result, including warnings and repair details.
        validation = __import__("app.models", fromlist=["ValidationResult"]).ValidationResult(**winner["validation"])
        repair_history = winner["repair_history"]
        self.curator.run(run_id, spec, selected, validation, repair_history)
        self.curator.record_candidates(run_id, candidate_results)
        event_log.append({"agent": "CuratorAgent", "status": "ok", "run_id": run_id, "knowledge_nodes": self.store.graph.number_of_nodes()})
        result = WorkflowResult(run_id=run_id, spec=spec, knowledge=knowledge, plans=plans, selected_plan=selected, generated_files=generated_files, validation=validation, repair_history=repair_history, candidate_results=candidate_results, llm_trace={"requirement": requirement_trace, "structured_advice": advisor_trace.to_dict() if advisor_trace else None}, search_trace=search_trace.to_dict(), event_log=event_log)
        json_path, md_path = write_report(result, self.settings.reports_dir)
        result.report_json, result.report_markdown = str(json_path), str(md_path)
        # Rewrite JSON after adding report paths.
        json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result
