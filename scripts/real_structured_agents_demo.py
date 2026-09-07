from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.advisor_agent import AdvisorAgent
from app.agents.requirement_agent import RequirementUnderstandingAgent
from app.knowledge.retriever import RetrieverAgent
from app.knowledge.store import KnowledgeStore
from app.llm.local_transformers import LocalTransformersLLM


def main() -> int:
    model_path = "/data/public_checkpoints/huggingface_models/Qwen2.5-14B-Instruct"
    llm = LocalTransformersLLM(model_path)
    description = "根据客户年龄、地区、登录频率、消费金额和投诉次数预测客户是否流失，表格数据类别不平衡，要求 ROC-AUC 不低于 0.75，并输出概率，优先可解释方案。"
    data_path = PROJECT_ROOT / "data/churn_demo.csv"
    started = time.perf_counter()
    spec, requirement_trace = RequirementUnderstandingAgent(llm, "local_instruction").run(description, data_path)
    with tempfile.TemporaryDirectory(prefix="structured_agents_") as temp:
        store = KnowledgeStore(Path(temp) / "knowledge.sqlite", Path(temp) / "knowledge.graphml")
        store.seed_from_json(PROJECT_ROOT / "app/knowledge/seed_data/knowledge.json")
        store.ensure_catalog_nodes()
        store.add_validation_run({"run_id": "evidence_run_084", "capability_id": "cap_churn_prediction_v1", "algorithm_id": "algorithm_logistic_regression", "status": "passed", "task_type": "binary_classification", "domain": "customer_churn", "target": "churn", "feature_columns": spec.feature_columns, "metrics": {"roc_auc": 0.84}, "runtime_seconds": 1.1})
        knowledge = RetrieverAgent(store).run(spec)
        advice, planner_trace = AdvisorAgent(llm, max_context_chars=10000).run(spec, knowledge)
    success = requirement_trace.get("status") == "ok" and planner_trace.get("status") == "ok"
    evidence = {"model": Path(model_path).name, "success": success, "requirement": spec.to_dict(), "requirement_trace": requirement_trace, "planner_advice": advice, "planner_trace": planner_trace, "retrieval_trace": knowledge.retrieval_trace, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    target = PROJECT_ROOT / "docs/evidence/real_structured_agents.json"
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": success, "requirement_status": requirement_trace.get("status"), "planner_status": planner_trace.get("status"), "model": evidence["model"], "elapsed_seconds": evidence["elapsed_seconds"], "context": planner_trace.get("context")}, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

