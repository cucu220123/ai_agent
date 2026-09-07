"""Persist every measured candidate/repair version, not just the winner."""
from __future__ import annotations
import hashlib
import json
import platform
from importlib.metadata import version
from typing import Any
from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec, AlgorithmPlan, ValidationResult
from app.llm.security import sanitize


def identifier(prefix: str, payload: Any) -> str:
    return prefix + "_" + hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


class CuratorAgent:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def curate(self, run_id: str, spec: CapabilitySpec, candidates: list[dict], winner_id: str) -> dict[str, Any]:
        capability_id = identifier("capability", [spec.domain, spec.task_type, spec.target_column])
        capability_version = identifier("schema", [spec.input_schema, spec.output_schema])
        self.store.upsert_capability({"id": capability_id, "name": spec.capability_name, "domain": spec.domain, "task_type": spec.task_type, "target": spec.target_column, "capability_version": capability_version, "input_schema": spec.input_schema, "output_schema": spec.output_schema, "origin": "measured_workflow"})
        task_id = f"task_{spec.task_type}"
        self.store.upsert_knowledge_item(task_id, "Task", {"name": spec.task_type, "task_type": spec.task_type})
        self.store.add_edge(capability_id, task_id, "SOLVES")
        environment = {"python": platform.python_version(), "dependencies": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "networkx", "pydantic")}}
        env_id = identifier("environment", environment)
        self.store.upsert_knowledge_item(env_id, "Environment", environment)
        recorded, versions, failures, repairs = [], [], [], []
        for candidate in candidates:
            plan = candidate["plan"]
            algorithm_id = plan.get("base_algorithm_id") or plan["algorithm_id"].split("__")[0]
            self.store.add_edge(capability_id, algorithm_id, "USES_ALGORITHM")
            self.store.add_edge(algorithm_id, task_id, "SUITABLE_FOR")
            config_id = identifier("config", [algorithm_id, plan["hyperparameters"], plan["preprocessing"]])
            self.store.upsert_knowledge_item(config_id, "HyperparameterConfig", {"algorithm_id": algorithm_id, "parameters": plan["hyperparameters"], "preprocessing": plan["preprocessing"], "variant": plan.get("config_variant")})
            for step in plan["preprocessing"]:
                step_id = identifier("preprocessing", step)
                self.store.upsert_knowledge_item(step_id, "PreprocessingStrategy", {"name": step})
                self.store.add_edge(algorithm_id, step_id, "USES_PREPROCESSING")
            attempts = candidate.get("attempts", [])
            for index, attempt in enumerate(attempts):
                validation = attempt["validation"]
                is_selected = plan["algorithm_id"] == winner_id and index == len(attempts) - 1
                measurement_id = run_id if is_selected else f"{run_id}:{plan['algorithm_id']}:v{index + 1}"
                profile = validation.get("dataset_profile") or spec.dataset_profile
                dataset_id = profile.get("dataset_id") or identifier("dataset", profile)
                self.store.upsert_knowledge_item(dataset_id, "Dataset", {**profile, "name": spec.dataset_path or dataset_id, "task_type": spec.task_type})
                self.store.add_edge(algorithm_id, dataset_id, "VALIDATED_ON")
                version_id = attempt["version_id"]
                self.store.add_algorithm_version({"id": version_id, "algorithm_id": algorithm_id, "version": index + 1, "parent_version": attempt.get("parent_version"), "code_hash": attempt["code_hash"], "code_source": attempt["code_source"], "algorithm_path": attempt["algorithm_path"], "status": validation["status"], "timestamp": self.store._now()})
                payload = {
                    "run_id": measurement_id, "workflow_run_id": run_id, "capability_id": capability_id, "capability_version": capability_version,
                    "algorithm_id": algorithm_id, "candidate_id": plan["algorithm_id"], "version_id": version_id,
                    "origin": "measured_workflow", "selected": is_selected, "status": validation["status"],
                    "success": validation["status"] == "passed", "task_type": spec.task_type, "domain": spec.domain,
                    "target": spec.target_column, "data_type": spec.data_type, "feature_columns": spec.feature_columns,
                    "feature_profile": spec.input_schema, "dataset_profile": profile, "dataset_id": dataset_id,
                    "sample_size": profile.get("rows"), "class_balance": profile.get("class_balance", {}),
                    "config_id": config_id, "hyperparameters": plan["hyperparameters"], "preprocessing": plan["preprocessing"],
                    "metrics": validation["metrics"], "checks": validation["checks"], "runtime_seconds": validation["runtime_seconds"],
                    "resource_usage": validation.get("resource_usage", {}), "environment": environment, "environment_id": env_id,
                    "failure_reason": validation.get("errors", []), "repair_history": candidate.get("repair_history", [])[:index],
                    "timestamp": self.store._now(), "code_hash": attempt["code_hash"], "provenance": {"algorithm_path": attempt["algorithm_path"], "workflow_run_id": run_id, "validation_round": index},
                }
                self.store.add_validation_run(sanitize(payload))
                self.store.add_edge(measurement_id, env_id, "REQUIRES")
                for constraint, check in validation["checks"].items():
                    if "passed" not in check: continue
                    constraint_id = identifier("constraint", [constraint, spec.metric_thresholds if constraint == "metrics" else spec.resource_constraints])
                    self.store.upsert_knowledge_item(constraint_id, "Constraint", {"name": constraint, "requirement": spec.metric_thresholds if constraint == "metrics" else spec.resource_constraints})
                    if check["passed"]:
                        self.store.add_edge(measurement_id, constraint_id, "SATISFIES")
                recorded.append(measurement_id)
                versions.append(version_id)
                if validation["status"] != "passed":
                    diagnosis = attempt.get("diagnosis", {})
                    repair = candidate.get("repair_history", [])[index] if index < len(candidate.get("repair_history", [])) else {}
                    repaired = index + 1 < len(attempts) and attempts[index + 1]["validation"]["status"] == "passed"
                    failure_id = f"failure_{version_id}"
                    experience = {"id": failure_id, "kind": "failure", "algorithm_id": algorithm_id, "run_id": measurement_id, "task_type": spec.task_type, "domain": spec.domain, "data_type": spec.data_type, "target": spec.target_column, "feature_columns": spec.feature_columns, "dataset_profile": profile, "failure_type": validation.get("failure_type"), "root_cause": diagnosis.get("root_cause") or validation.get("root_cause"), "triggering_condition": diagnosis.get("triggering_condition"), "observed_error": validation["errors"], "repair_action": repair.get("strategy", repair.get("changes", [])), "repair_success": repaired, "reusable_lesson": diagnosis.get("reusable_lesson"), "repair_history": [repair] if repair else [], "timestamp": self.store._now(), "provenance": {"validation_run": measurement_id, "code_hash": attempt["code_hash"]}}
                    self.store.add_experience(sanitize(experience))
                    failures.append(failure_id)
                    if repair:
                        repair_id = f"repair_{version_id}"
                        self.store.add_repair_experience(sanitize({"id": repair_id, "failure_id": failure_id, "algorithm_id": algorithm_id, "task_type": spec.task_type, "strategy": repair.get("strategy", repair.get("changes", [])), "success": repaired, "provider": repair.get("provider"), "from_version": version_id, "to_version": attempts[index + 1]["version_id"] if index + 1 < len(attempts) else None, "timestamp": self.store._now()}))
                        repairs.append(repair_id)
        self.store.export_graph()
        return {"workflow_run_id": run_id, "validation_run_ids": recorded, "algorithm_version_ids": versions, "failure_experience_ids": failures, "repair_experience_ids": repairs, "capability_id": capability_id}

    def run(self, run_id: str, spec: CapabilitySpec, plan: AlgorithmPlan, result: ValidationResult, repair_history: list[dict]) -> None:
        # Compatibility for external users of the previous simple curator entry point.
        self.store.add_validation_run({"run_id": run_id, "algorithm_id": plan.base_algorithm_id or plan.algorithm_id, "task_type": spec.task_type, "domain": spec.domain, "target": spec.target_column, "feature_columns": spec.feature_columns, "status": result.status, "metrics": result.metrics, "dataset_profile": result.dataset_profile, "repair_history": repair_history, "timestamp": self.store._now()})

    def record_candidates(self, run_id: str, candidate_results: list[dict]) -> None:
        raise RuntimeError("Use curate() with spec and all attempt histories; incomplete candidate persistence is no longer supported")

