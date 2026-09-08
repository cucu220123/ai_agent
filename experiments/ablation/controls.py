"""Experiment-scoped adapters around the existing workflow, never a second workflow.

All controls default to enabled. The normal application imports none of this
module. Each trial owns a separate store, validator adapter and API clients.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json

from app.agents.planning_context import PlanningContextBuilder
from app.models import KnowledgeContext
from app.plugins.registry import DEFAULT_REGISTRY
from app.search.beam import SearchTrace
from app.llm.security import sanitize


@dataclass(frozen=True)
class Controls:
    enable_graph_retrieval: bool = True
    enable_semantic_retrieval: bool = True
    enable_experience_retrieval: bool = True
    enable_beam_search: bool = True
    enable_multi_candidate: bool = True
    enable_repair: bool = True
    enable_knowledge_context: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def empty_graph() -> dict[str, Any]:
    return {"anchors": [], "nodes": [], "edges": [], "paths": [],
            "serialized": {"candidate_algorithms": []}, "disabled": True}


class DisabledGraph:
    def retrieve(self, *args, **kwargs):
        return empty_graph()


class DisabledDocuments:
    last_backend = "disabled"
    last_error = None

    def retrieve(self, *args, **kwargs):
        return []


class CatalogOnly:
    """Only executable IDs/names are available; no retrieved domain material."""

    def run(self, spec):
        catalog = [{"id": f"algorithm_{p.id}", "name": p.name,
                    "task_types": p.task_types, "origin": "execution_contract"}
                   for p in DEFAULT_REGISTRY.algorithms_for(spec.task_type)]
        return KnowledgeContext(algorithms=catalog, graph_evidence=empty_graph(),
                                retrieval_trace={"strategy": "disabled", "graph_nodes": 0,
                                "graph_edges": 0, "semantic_documents": 0, "historical_cases": 0})


class RequirementReference(PlanningContextBuilder):
    """The real Planner requires a citation; cite current input, not invented knowledge.

This does not add a retrieved source. Retrieval/citation metrics exclude this ID.
"""

    def build(self, spec, knowledge):
        context, trace = super().build(spec, knowledge)
        if not trace["final_evidence_ids"]:
            context["current_request_reference"] = {"id": "current_user_requirement",
                                                     "raw_description": spec.raw_description}
            trace["final_evidence_ids"] = ["current_user_requirement"]
            trace["final_prompt_chars"] = len(json.dumps(context, ensure_ascii=False, separators=(",", ":")))
            trace["estimated_prompt_tokens"] = max(1, trace["final_prompt_chars"] // 3)
        return context, trace


class Capture:
    """Capture original component outputs without changing their behavior."""

    def __init__(self, component):
        self.component = component
        self.last = None

    def run(self, *args, **kwargs):
        self.last = self.component.run(*args, **kwargs)
        return self.last


class SearchControl:
    def __init__(self, original, controls: Controls):
        self.original, self.controls = original, controls
        self.parents = []
        self.last_trace = None

    def search(self, plans, spec, beam_width=3):
        self.parents = [plan.to_dict() for plan in plans]
        eligible = plans if self.controls.enable_multi_candidate else plans[:1]
        width = beam_width if self.controls.enable_multi_candidate else 1
        if self.controls.enable_beam_search:
            selected, trace = self.original.search(eligible, spec, width)
        else:
            # Original Planner Top-K: keep LLM parameter proposals, no state expansion.
            selected = eligible[:width]
            trace = SearchTrace("planner_top_k_without_state_expansion", width, 0,
                                [p.algorithm_id for p in selected],
                                {p.algorithm_id: p.search_score for p in plans})
        self.last_trace = trace.to_dict()
        return selected, trace


class DevelopmentValidator:
    """Reuse trusted validation on a fixed development-only split for every candidate."""

    def __init__(self, original, validation_path: Path, split_seed: int, thresholds: dict):
        self.original, self.validation_path = original, validation_path
        self.split_seed, self.thresholds = split_seed, thresholds

    def run(self, path, data, spec, *args, **kwargs):
        # The experiment's precommitted thresholds cannot be relaxed by an LLM.
        if spec.metric_thresholds != self.thresholds:
            raise ValueError("requirement changed preregistered thresholds")
        result = self.original.run(path, data, spec, *args,
                                   evaluation_data_path=self.validation_path, **kwargs)
        if "evaluation_split" in result.checks:
            result.checks["evaluation_split"].update(mode="ablation_development_holdout",
                                                    split_seed=self.split_seed,
                                                    independent_final_test=False)
        return result


class DisabledCritique:
    """No model/diagnosis when runtime repair is disabled; executor exits immediately."""

    def run(self, *args, **kwargs):
        return {"status": "disabled_by_ablation", "root_cause": None,
                "note": "No Critic model, Repair or revalidation was executed."}


class RecordCandidates:
    def __init__(self, original):
        self.original = original
        self.results = []
        self.started = []

    def run(self, *args, **kwargs):
        self.started.append(args[1].algorithm_id)
        result = self.original.run(*args, **kwargs)
        self.results.append(result)
        return result


class RecordingLLM:
    """Preserve sanitized exact prompts/responses, including every rejected output."""

    def __init__(self, original, directory: Path):
        self.original, self.directory = original, directory
        directory.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def __getattr__(self, key):
        return getattr(self.original, key)

    def complete(self, system, user, purpose="general", generation_config=None):
        self.count += 1
        record = {"purpose": purpose, "system": system, "user": user,
                  "generation_config": generation_config}
        try:
            result = self.original.complete(system, user, purpose, generation_config)
            record.update(status="ok", response=result)
            return result
        except Exception as exc:
            record.update(status="error", error=str(exc))
            raise
        finally:
            (self.directory / f"{self.count:03d}_{purpose}.json").write_text(
                json.dumps(sanitize(record), ensure_ascii=False, indent=2), encoding="utf-8")


def attach_controls(workflow, controls: Controls, validation: Path, seed: int,
                    thresholds: dict, raw_calls: Path):
    """Wire existing objects in one private trial; production imports/defaults untouched."""
    recorder = RecordingLLM(workflow.llm, raw_calls)
    workflow.llm = recorder
    for component in (workflow.requirement_agent, workflow.advisor, workflow.generator,
                      workflow.critic, workflow.repair, workflow.explainer):
        component.llm = recorder
    router = recorder.original.provider
    api_clients = [router.instruction, router.coder] if hasattr(router, "instruction") else [router]
    for backend in api_clients:
        if not hasattr(backend, "client"):
            continue  # Explicit fixture path is only used by automated tests.
        original_create = backend.client.chat.completions.create
        def seeded_create(*args, _create=original_create, **kwargs):
            return _create(*args, seed=seed, **kwargs)
        backend.client.chat.completions.create = seeded_create
    if not controls.enable_knowledge_context:
        workflow.retriever = CatalogOnly()
    else:
        if not controls.enable_graph_retrieval:
            workflow.retriever.graph = DisabledGraph()
        if not controls.enable_semantic_retrieval:
            workflow.retriever.semantic = DisabledDocuments()
    workflow.advisor.context_builder = RequirementReference(workflow.settings.planning_context_max_chars)
    workflow.retriever = Capture(workflow.retriever)
    workflow.planner = Capture(workflow.planner)
    workflow.searcher = SearchControl(workflow.searcher, controls)
    validator = DevelopmentValidator(workflow.validator, validation, seed, thresholds)
    workflow.validator = validator
    workflow.executor.validator = validator
    if not controls.enable_repair:
        workflow.executor.critic = DisabledCritique()
    workflow.executor = RecordCandidates(workflow.executor)
    return workflow


def freeze_experience_clock(as_of: str):
    """Freeze only prior recency calculation, not log/measurement timestamps."""
    import app.experience.retriever as module
    fixed = datetime.fromisoformat(as_of)
    class ReferenceDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)
    module.datetime = ReferenceDatetime
