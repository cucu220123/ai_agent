from __future__ import annotations

from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec, AlgorithmPlan, ValidationResult


class CuratorAgent:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def run(self, run_id: str, spec: CapabilitySpec, plan: AlgorithmPlan, result: ValidationResult, repair_history: list[dict]) -> None:
        self.store.add_validation_run({
            "run_id": run_id,
            "capability_id": "cap_churn_prediction_v1" if spec.target_column == "churn" else f"cap_{spec.target_column}",
            "algorithm_id": plan.algorithm_id,
            "status": result.status,
            "metrics": result.metrics,
            "checks": result.checks,
            "runtime_seconds": result.runtime_seconds,
            "repair_round": result.repair_round,
        })
        if result.status != "passed" or repair_history:
            self.store.add_experience({
                "id": f"experience_{run_id}",
                "algorithm_id": plan.algorithm_id,
                "kind": "success" if result.status == "passed" else "failure",
                "summary": "; ".join(result.errors) if result.errors else "算法通过验证",
                "metrics": result.metrics,
                "repair_history": repair_history,
            })

