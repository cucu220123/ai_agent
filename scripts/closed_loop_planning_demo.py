from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.planner_agent import PlannerAgent
from app.knowledge.retriever import RetrieverAgent
from app.knowledge.store import KnowledgeStore
from app.models import CapabilitySpec
from app.search.beam import BeamSearchPlanner


def snapshot(store: KnowledgeStore, spec: CapabilitySpec) -> dict:
    context = RetrieverAgent(store).run(spec)
    plans = PlannerAgent().run(spec, context)
    finalists, trace = BeamSearchPlanner().search(plans, spec, beam_width=3)
    return {
        "planning_priors": {plan.base_algorithm_id: plan.search_score for plan in plans},
        "beam_ranking": [plan.base_algorithm_id for plan in finalists],
        "beam_scores": {plan.algorithm_id: plan.search_score for plan in finalists},
        "historical_run_ids": [run.get("run_id") for run in context.historical_cases],
        "retrieval_trace": context.retrieval_trace,
        "search_trace": trace.to_dict(),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="closed_loop_planning_") as temp:
        temp_path = Path(temp)
        store = KnowledgeStore(temp_path / "knowledge.sqlite", temp_path / "knowledge.graphml")
        store.seed_from_json(PROJECT_ROOT / "app/knowledge/seed_data/knowledge.json")
        store.ensure_catalog_nodes()
        spec = CapabilitySpec(raw_description="客户流失预测", domain="customer_churn", capability_name="客户流失预测", target_column="churn", feature_columns=["age", "login_count_30d"], metrics=["roc_auc"], metric_thresholds={"roc_auc": 0.8})
        before = snapshot(store, spec)
        run_id = "closed_loop_high_similarity_run"
        store.add_validation_run({"run_id": run_id, "capability_id": "cap_churn_prediction_v1", "algorithm_id": "algorithm_logistic_regression", "status": "passed", "task_type": "binary_classification", "domain": "customer_churn", "target": "churn", "feature_columns": ["age", "login_count_30d"], "data_type": "tabular", "metrics": {"roc_auc": 0.99}, "runtime_seconds": 0.5, "timestamp": "2026-09-07T00:00:00Z"})
        after = snapshot(store, spec)
        payload = {
            "written_validation_run": run_id,
            "before": before,
            "after": after,
            "evidence": {
                "retrieved_after_writeback": run_id in after["historical_run_ids"],
                "logistic_prior_before": before["planning_priors"]["algorithm_logistic_regression"],
                "logistic_prior_after": after["planning_priors"]["algorithm_logistic_regression"],
                "ranking_before": before["beam_ranking"],
                "ranking_after": after["beam_ranking"],
                "planning_state_changed": before["planning_priors"] != after["planning_priors"],
            },
        }
        output = PROJECT_ROOT / "docs/evidence/closed_loop_planning.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload["evidence"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

