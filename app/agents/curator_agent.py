from __future__ import annotations

from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec, AlgorithmPlan, ValidationResult


class CuratorAgent:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def run(self, run_id: str, spec: CapabilitySpec, plan: AlgorithmPlan, result: ValidationResult, repair_history: list[dict]) -> None:
        algorithm_id = plan.base_algorithm_id or plan.algorithm_id
        config_id = f"config_{run_id}"
        version_id = f"version_{run_id}"
        self.store.upsert_knowledge_item(config_id, "HyperparameterConfig", {"id": config_id, "algorithm_id": algorithm_id, "parameters": plan.hyperparameters, "preprocessing_variant": plan.preprocessing_variant, "config_variant": plan.config_variant})
        self.store.add_algorithm_version({"id": version_id, "algorithm_id": algorithm_id, "version": f"{run_id}.v{result.repair_round + 1}", "code_hash": result.resource_usage.get("code_hash", ""), "status": result.status, "timestamp": self.store._now()})
        self.store.add_validation_run({
            "run_id": run_id,
            "capability_id": "cap_churn_prediction_v1" if spec.target_column == "churn" else f"cap_{spec.target_column}",
            "algorithm_id": algorithm_id,
            "status": result.status,
            "metrics": result.metrics,
            "checks": result.checks,
            "runtime_seconds": result.runtime_seconds,
            "repair_round": result.repair_round,
            "task_type": spec.task_type,
            "domain": spec.domain,
            "target": spec.target_column,
            "feature_columns": spec.feature_columns,
            "data_type": spec.data_type,
            "dataset_profile": result.dataset_profile,
            "resource_usage": result.resource_usage,
            "timestamp": self.store._now(),
            "dataset_id": "dataset_churn_demo" if spec.target_column == "churn" else None,
            "config_id": config_id,
        })
        if result.status != "passed" or repair_history:
            failure_id = f"experience_{run_id}"
            self.store.add_experience({
                "id": failure_id,
                "algorithm_id": algorithm_id,
                "kind": "success" if result.status == "passed" else "failure",
                "summary": "; ".join(result.errors) if result.errors else "算法通过验证",
                "metrics": result.metrics,
                "repair_history": repair_history,
                "run_id": run_id,
                "failure_type": result.failure_type or "validation_failure",
                "root_cause": result.root_cause or (result.errors[0] if result.errors else ""),
            })
            for index, repair in enumerate(repair_history, 1):
                self.store.add_repair_experience({"id": f"repair_{run_id}_{index}", "failure_id": failure_id, "strategy": repair.get("changes", []), "diagnosis": repair.get("diagnosis", {}), "success": result.status == "passed"})

    def record_candidates(self, run_id: str, candidate_results: list[dict]) -> None:
        """Persist failed alternatives and repaired candidate experience."""
        for item in candidate_results:
            validation = item.get("validation", {})
            plan = item.get("plan", {})
            if validation.get("status") == "passed" and not item.get("repair_history"):
                continue
            self.store.add_experience({
                "id": f"experience_{run_id}_{plan.get('algorithm_id', 'unknown')}",
                "algorithm_id": plan.get("algorithm_id"),
                "kind": "candidate_failure" if validation.get("status") != "passed" else "candidate_repaired",
                "summary": "; ".join(validation.get("errors", [])) or "candidate required repair",
                "metrics": validation.get("metrics", {}),
                "repair_history": item.get("repair_history", []),
                "run_id": run_id,
                "failure_type": validation.get("failure_type", "candidate_failure"),
                "root_cause": "; ".join(validation.get("errors", [])),
            })
