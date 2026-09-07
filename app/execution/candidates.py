from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from app.agents.generator_agent import GeneratorAgent
from app.agents.critic_agent import CriticAgent
from app.agents.repair_agent import RepairAgent
from app.config import Settings
from app.models import AlgorithmPlan, CapabilitySpec, KnowledgeContext
from app.validation.runner import ValidationRunner
from app.agents.protocol import AgentRuntime
from app.generation.task_contracts import build_codegen_prompt


class CandidateExecutor:
    """Generate -> immutable snapshot -> validate -> critic -> repair -> revalidate."""

    def __init__(self, settings: Settings, generator: GeneratorAgent, validator: ValidationRunner, critic: CriticAgent, repair: RepairAgent):
        self.settings, self.generator, self.validator, self.critic, self.repair = settings, generator, validator, critic, repair

    def run(self, run_id: str, plan: AlgorithmPlan, spec: CapabilitySpec, knowledge: KnowledgeContext, data: Path, runtime: AgentRuntime, *, inject_failure: bool = False) -> dict[str, Any]:
        emit = runtime.emit
        directory = self.settings.generated_dir / run_id / plan.algorithm_id
        path = runtime.call("CoderAgent", "generate", self.generator.run, directory, spec, plan, mode=self.settings.codegen_mode, knowledge=knowledge.planning_context, allow_template_fallback=not self.settings.strict_real_llm)
        metadata = json.loads(path.with_name("algorithm_meta.json").read_text())
        generation = metadata["generation"]
        code_source = "llm" if generation["status"] == "llm_code_accepted" else generation["status"]
        emit("CoderAgent", generation["status"], algorithm=plan.algorithm_id, generation=generation)
        if inject_failure:
            original = path.read_text()
            broken = original.replace("def predict(", "def predict_broken(", 1)
            if broken == original:
                raise ValueError("repair demo injection requires an existing predict function")
            path.write_text(broken)
            emit("DemoFaultInjection", "applied", reason="rename predict to create a reproducible interface failure")
        attempts, repairs = [], []
        parent = None
        for round_no in range(self.settings.max_repair_rounds + 1):
            version = f"{run_id}_{plan.algorithm_id}_v{round_no + 1}"
            snapshot = directory / "versions" / f"v{round_no + 1}.py"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            source = path.read_text()
            snapshot.write_text(source)
            digest = hashlib.sha256(source.encode()).hexdigest()
            emit("Sandbox", "started", algorithm=plan.algorithm_id, version_id=version)
            validation = runtime.call("ValidatorAgent", "validate", self.validator.run, snapshot, data, spec, plan.algorithm_name, round_no, plan.hyperparameters)
            validation.resource_usage["code_hash"] = digest
            attempt: dict[str, Any] = {"version_id": version, "parent_version": parent, "code_hash": digest, "algorithm_path": str(snapshot), "code_source": code_source, "validation": validation.to_dict()}
            attempts.append(attempt)
            emit("ValidatorAgent", validation.status, algorithm=plan.algorithm_id, round=round_no, version_id=version, metrics=validation.metrics, errors=validation.errors[:3])
            if validation.status == "passed":
                break
            diagnosis = runtime.call("CriticAgent", "critique", self.critic.run, spec, plan, validation, source, knowledge.experiences)
            attempt["diagnosis"] = diagnosis
            emit("CriticAgent", diagnosis.get("status", "fallback"), algorithm=plan.algorithm_id, diagnosis=diagnosis)
            if round_no == self.settings.max_repair_rounds:
                break
            feedback = json.dumps({"validation": validation.to_dict(), "diagnosis": diagnosis, "metric_thresholds": spec.metric_thresholds, "resource_constraints": spec.resource_constraints}, ensure_ascii=False)
            repair = runtime.call("RepairAgent", "repair", self.repair.repair, path, feedback, round_no + 1, knowledge.experiences, spec.task_type, spec.target_column, context={"requirement": spec.to_dict(), "execution_plan": plan.to_dict(), "host_contract": build_codegen_prompt(spec, plan)})
            repair["diagnosis"] = diagnosis
            repair["from_version"] = version
            repairs.append(repair)
            emit("RepairAgent", repair.get("status", "unknown"), algorithm=plan.algorithm_id, round=round_no + 1)
            if repair.get("after_sha256") == digest:
                break
            if repair.get("status") == "llm_repair_accepted":
                code_source = "repaired_llm"
            parent = version
        metadata.update(final_code_sha256=attempts[-1]["code_hash"], final_version=attempts[-1]["version_id"], validation_status=validation.status)
        path.with_name("algorithm_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        return {"plan": plan.to_dict(), "validation": validation.to_dict(), "repair_history": repairs, "algorithm_path": attempts[-1]["algorithm_path"], "artifact_sha256": attempts[-1]["code_hash"], "code_source": code_source, "generation_trace": generation, "attempts": attempts}
