"""End-to-end orchestration; candidate repair/execution is a separate component."""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from app.agents.advisor_agent import AdvisorAgent
from app.agents.critic_agent import CriticAgent
from app.agents.curator_agent import CuratorAgent
from app.agents.explanation_agent import ExplanationAgent
from app.agents.generator_agent import GeneratorAgent
from app.agents.parser_agent import ParserAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.protocol import AgentRuntime
from app.agents.repair_agent import RepairAgent
from app.agents.requirement_agent import RequirementUnderstandingAgent
from app.config import Settings, get_settings
from app.execution.candidates import CandidateExecutor
from app.knowledge.bootstrap import KnowledgeBootstrapper
from app.knowledge.retriever import RetrieverAgent
from app.knowledge.store import KnowledgeStore
from app.llm.factory import build_llm
from app.llm.security import sanitize
from app.metrics.registry import METRIC_REGISTRY
from app.models import WorkflowResult, ValidationResult
from app.search.beam import BeamSearchPlanner
from app.validation.dataset import profile_dataset
from app.validation.report import write_report
from app.validation.runner import ValidationRunner


class AlgorithmFactoryWorkflow:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.store = KnowledgeStore(self.settings.knowledge_db, self.settings.graphml_path)
        if not self.store.list_capabilities():
            self.store.seed_from_json(Path(__file__).parent / "knowledge/seed_data/knowledge.json")
        self.store.ensure_catalog_nodes()
        self.parser = ParserAgent()  # Explicit offline compatibility API.
        self.llm = build_llm(self.settings)
        provider = self.settings.llm_provider
        self.requirement_agent = RequirementUnderstandingAgent(self.llm, provider)
        self.retriever = RetrieverAgent(self.store, self.settings.embedding_model_path)
        self.advisor = AdvisorAgent(self.llm, self.settings.planning_context_max_chars)
        self.planner = PlannerAgent()
        self.searcher = BeamSearchPlanner()
        self.generator = GeneratorAgent(self.llm, provider)
        self.critic = CriticAgent(self.llm, provider)
        self.repair = RepairAgent(self.llm, provider)
        self.validator = ValidationRunner(self.settings.validation_timeout_seconds, self.settings.validation_memory_mb, self.settings.validation_seed_variance, self.settings.validation_repeats, self.settings.validation_cv_folds)
        self.curator = CuratorAgent(self.store)
        self.explainer = ExplanationAgent(self.llm, provider)
        self.executor = CandidateExecutor(self.settings, self.generator, self.validator, self.critic, self.repair)

    def run(self, description: str, data_path: str | Path, provider_note: str | None = None, *, source_paths: list[Path] | None = None, inject_repair_failure: bool = False) -> WorkflowResult:
        run_id = uuid.uuid4().hex[:12]
        events = []
        runtime = AgentRuntime(events)
        start_call = len(self.llm.calls)
        strict = self.settings.strict_real_llm and self.settings.llm_provider != "mock"
        data = Path(data_path).resolve()
        try:
            spec, requirement_trace = runtime.call("RequirementAgent", "understand", self.requirement_agent.run, description, data)
            if strict and requirement_trace["status"] != "ok":
                raise RuntimeError("Real requirement understanding failed; see explicit trace")
            spec.dataset_profile = profile_dataset(data, spec.target_column)
            if "positive_rate" in spec.dataset_profile:
                spec.class_imbalance.update(positive_rate=spec.dataset_profile["positive_rate"], is_imbalanced=spec.dataset_profile["minority_rate"] < .25)
            # Source extraction enters the same workflow and is cached by source hash.
            default_sources = [self.settings.project_root / "data/business_material.md", self.settings.project_root / "data/text_material.md", self.settings.project_root / "data/reference_preprocessing.py"]
            extraction = runtime.call("KnowledgeExtractionAgent", "ingest", KnowledgeBootstrapper(self.store, self.llm, self.settings.llm_provider).run, source_paths if source_paths is not None else default_sources) if self.settings.bootstrap_knowledge else []
            if strict and any(item["extraction_status"] != "ok" for item in extraction):
                raise RuntimeError("Real knowledge extraction failed; refusing a template-only knowledge demo")
            self.store.ensure_catalog_nodes()
            knowledge = runtime.call("RetrievalAgent", "retrieve", self.retriever.run, spec)
            runtime.emit("GraphSearch", "ok", anchors=knowledge.graph_evidence["anchors"], nodes=len(knowledge.graph_evidence["nodes"]), edges=len(knowledge.graph_evidence["edges"]), historical_run_ids=[x["run_id"] for x in knowledge.historical_cases])
            advice, advisor_trace = runtime.call("PlannerAgent", "advise", self.advisor.run, spec, knowledge)
            knowledge.planning_context = advisor_trace["final_prompt_evidence"]
            knowledge.planner_advice = advice if advisor_trace["status"] == "ok" else {}
            if strict and advisor_trace["status"] != "ok":
                raise RuntimeError("Real LLM planning failed; refusing to call deterministic advice real planning")
            # LLM recommendations shape proposals, but cannot erase compatible exploration.
            parents = runtime.call("PlannerAgent", "plan", self.planner.run, spec, knowledge)
            if not parents: raise ValueError(f"no compatible executable plugin for {spec.task_type}")
            plans, search = runtime.call("PlannerAgent", "search", self.searcher.search, parents, spec, self.settings.beam_width)
            candidates = [self.executor.run(run_id, plan, spec, knowledge, data, runtime, inject_failure=inject_repair_failure and index == 0) for index, plan in enumerate(plans)]
            eligible = [item for item in candidates if item["validation"]["status"] == "passed" and (not strict or item["code_source"] in {"llm", "repaired_llm"})]
            ranking = eligible or candidates
            ranking = sorted(ranking, key=lambda item: (METRIC_REGISTRY.ranking_score(spec.task_type, item["validation"]["metrics"], item["validation"]["runtime_seconds"], spec.metrics, bool(spec.target_column))[0], -item["validation"]["runtime_seconds"]), reverse=True)
            winner = ranking[0]
            winner_id = winner["plan"]["algorithm_id"]
            selected = next(plan for plan in plans if plan.algorithm_id == winner_id)
            writeback = runtime.call("CuratorAgent", "curate", self.curator.curate, run_id, spec, candidates, winner_id)
            explanation = runtime.call("ExplanationAgent", "explain", self.explainer.run, spec, knowledge, candidates, winner)
            result = WorkflowResult(run_id=run_id, spec=spec, knowledge=knowledge, plans=plans, selected_plan=selected, generated_files=[attempt["algorithm_path"] for candidate in candidates for attempt in candidate["attempts"]], validation=ValidationResult(**winner["validation"]), repair_history=winner["repair_history"], candidate_results=candidates, llm_trace={"requirement": requirement_trace, "structured_advice": advisor_trace, "calls": self.llm.calls[start_call:], "strict_real_mode": strict, "real_winner": bool(eligible) and winner["code_source"] in {"llm", "repaired_llm"}, "provider_note": provider_note}, search_trace=search.to_dict(), event_log=events, extraction_trace=extraction, explanation=explanation, writeback=writeback)
            json_path, md_path = write_report(result, self.settings.reports_dir)
            result.report_json, result.report_markdown = str(json_path), str(md_path)
            json_path.write_text(json.dumps(sanitize(result.to_dict()), ensure_ascii=False, indent=2))
            if strict and not eligible:
                raise RuntimeError(f"No passing real-LLM candidate. Failure report retained: {json_path}")
            return result
        except Exception as exc:
            failure = {"run_id": run_id, "status": "failed", "error": str(exc), "event_log": events, "llm_calls": self.llm.calls[start_call:], "requirement_trace": locals().get("requirement_trace"), "extraction_trace": locals().get("extraction"), "advisor_trace": locals().get("advisor_trace")}
            self.settings.reports_dir.joinpath(f"{run_id}.failure.json").write_text(json.dumps(sanitize(failure), ensure_ascii=False, indent=2))
            raise

