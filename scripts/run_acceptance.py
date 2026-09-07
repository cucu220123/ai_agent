"""Real-model acceptance with resumable stages and assertions over saved evidence.

No provider substitution is performed here. Cloud failure must be recorded with
probe_llm before explicitly choosing a local real-model endpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings
from app.knowledge.extractor import CapabilityExtractor
from app.llm.security import sanitize
from app.workflow import AlgorithmFactoryWorkflow
from scripts.generate_demo_data import generate


ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2), encoding="utf-8")


def assert_real(result: dict) -> None:
    assert result["validation"]["status"] == "passed"
    assert result["llm_trace"]["strict_real_mode"] and result["llm_trace"]["real_winner"]
    assert result["llm_trace"]["requirement"]["status"] == "ok"
    assert result["llm_trace"]["structured_advice"]["status"] == "ok"
    assert result["explanation"]["status"] == "ok", result["explanation"].get("error")
    calls = result["llm_trace"]["calls"]
    assert calls and all("mock" not in str(call.get("provider", "")).lower() for call in calls)
    # Rejected real model candidates are legitimate comparison evidence;
    # they must remain failed and cannot be silently turned into templates.
    assert all(candidate["code_source"] in {"llm", "repaired_llm", "llm_code_rejected"} for candidate in result["candidate_results"])


def run(output: Path, stage: str, provider: str) -> dict:
    output = output.resolve()
    work = output / "workspace"
    work.mkdir(parents=True, exist_ok=True)
    (work / "data").mkdir(exist_ok=True)
    for name in ("business_material.md", "text_material.md", "reference_preprocessing.py", "text_demo.csv"):
        shutil.copy2(ROOT / "data" / name, work / "data" / name)
    base = get_settings()
    settings = replace(base, project_root=work, data_dir=work / "data", generated_dir=work / "generated", reports_dir=work / "reports", knowledge_db=work / "knowledge.sqlite", graphml_path=work / "knowledge.graphml", llm_provider=provider, strict_real_llm=True, beam_width=3, max_repair_rounds=3, planning_context_max_chars=18000)
    workflow = AlgorithmFactoryWorkflow(settings)
    stages = [stage] if stage != "all" else ["first", "second", "repair", "cross", "verify"]
    description = "Build customer churn prediction for mixed numeric/categorical tabular customer data. Target column churn, binary 0/1. Exclude churn from features. Require ROC-AUC >= 0.80 and positive-class probability. Report F1, precision, recall; handle missing values, unseen categories and class imbalance. Compare logistic regression, random forest and gradient boosting."
    for current in stages:
        destination = output / f"{current}.json"
        if current != "verify" and destination.exists():
            print(f"[Acceptance] retained completed stage {current}", flush=True)
            continue
        if current == "first":
            # Recover an already written complete report after an interrupted
            # acceptance process. Validate it without changing its run lineage.
            existing = []
            for path in settings.reports_dir.glob("*.json"):
                candidate = json.loads(path.read_text())
                if candidate.get("validation", {}).get("status") == "passed" and not candidate.get("knowledge", {}).get("historical_cases", []):
                    existing.append(candidate)
            if existing:
                result = existing[-1]
                assert_real(result)
                save(destination, result)
                shutil.copy2(settings.graphml_path, output / "graph_after_first.graphml")
                print("[Acceptance] recovered verified completed first report", flush=True)
                continue
            assert not workflow.store.list_validation_runs(1000), "First stage needs a fresh knowledge workspace"
            data = generate(work / "data/churn_first.csv", 1200, 42)
            result = workflow.run(description, data, provider_note="Explicit real-model API; see cloud_probe.json for original endpoint availability").to_dict()
            assert_real(result)
            assert not result["knowledge"]["historical_cases"]
            save(destination, result)
            shutil.copy2(settings.graphml_path, output / "graph_after_first.graphml")
        elif current == "second":
            first = json.loads((output / "first.json").read_text())
            measured = {"run_id": first["run_id"], "algorithm": first["selected_plan"]["algorithm_name"], "task_type": first["spec"]["task_type"], "dataset": first["spec"]["dataset_profile"], "config": first["selected_plan"]["hyperparameters"], "validation": {key: first["validation"][key] for key in ("status", "metrics", "runtime_seconds", "resource_usage")}, "provenance": {"source": "first.json", "workflow_run_id": first["run_id"]}}
            report = work / "data/measured_experiment.json"
            save(report, measured)
            extraction = CapabilityExtractor(workflow.llm, provider).ingest(report, workflow.store)
            save(output / "report_extraction.json", extraction)
            assert extraction["extraction_trace"]["status"] == "ok"
            data = generate(work / "data/churn_second.csv", 1050, 2026)
            result = workflow.run(description + " This is a new customer cohort; prefer interpretable models when their current metric is competitive. Prediction latency per row must be below 500 ms.", data).to_dict()
            assert_real(result)
            reference = first["run_id"]
            retrieved = [x["run_id"] for x in result["knowledge"]["historical_cases"]]
            context = result["llm_trace"]["structured_advice"]["final_prompt_evidence"]
            assert reference in retrieved
            assert reference in json.dumps(context)
            assert any(reference in p["evidence_ids"] for p in result["plans"])
            save(destination, result)
            save(output / "closed_loop_proof.json", {"prior_run_id": reference, "next_run_id": result["run_id"], "first_dataset": first["spec"]["dataset_profile"], "second_dataset": result["spec"]["dataset_profile"], "before_historical_run_ids": [x["run_id"] for x in first["knowledge"]["historical_cases"]], "after_historical_run_ids": retrieved, "exact_second_planner_context": context, "first_plans": first["plans"], "second_plans": result["plans"], "first_search": first["search_trace"], "second_search": result["search_trace"]})
        elif current == "repair":
            repair_workflow = AlgorithmFactoryWorkflow(replace(settings, beam_width=1))
            result = repair_workflow.run(description, work / "data/churn_first.csv", inject_repair_failure=True).to_dict()
            assert_real(result)
            candidate = result["candidate_results"][0]
            assert candidate["attempts"][0]["validation"]["status"] == "failed"
            assert candidate["attempts"][-1]["validation"]["status"] == "passed"
            assert candidate["attempts"][0]["code_hash"] != candidate["attempts"][-1]["code_hash"]
            assert any(r["status"] == "llm_repair_accepted" for r in candidate["repair_history"])
            assert candidate["attempts"][0]["diagnosis"]["status"] == "ok"
            save(destination, result)
            demo = output / "self_repair_demo"
            demo.mkdir(exist_ok=True)
            shutil.copy2(candidate["attempts"][0]["algorithm_path"], demo / "before.py")
            shutil.copy2(candidate["attempts"][-1]["algorithm_path"], demo / "after.py")
            save(demo / "proof.json", {"run_id": result["run_id"], "fault_injection": "explicit demo-only predict rename", "attempts": candidate["attempts"], "repair_history": candidate["repair_history"]})
            # A subsequent retrieval, using the same structured task, must reuse the failure.
            from app.models import CapabilitySpec
            knowledge = repair_workflow.retriever.run(CapabilitySpec(**result["spec"]))
            matching = [x for x in knowledge.experiences if x.get("workflow_run_id") == result["run_id"] or result["run_id"] in str(x.get("id", ""))]
            assert matching, "new failure experience was not retrievable"
            save(demo / "next_retrieval.json", {"experiences": matching, "retrieval_trace": knowledge.retrieval_trace})
        elif current == "cross":
            result = workflow.run("Classify English customer support text sentiment using column text and binary target label (0/1). This is text_classification, not tabular classification. Use TF-IDF with logistic regression, compare configurations. Report accuracy and weighted F1. Output prediction only; no probability requirement. Handle empty or missing text and unseen vocabulary. No minimum metric threshold.", work / "data/text_demo.csv").to_dict()
            assert_real(result)
            assert result["spec"]["task_type"] == "text_classification"
            save(destination, result)
        elif current == "verify":
            reports = {name: json.loads((output / f"{name}.json").read_text()) for name in ("first", "second", "repair", "cross")}
            for result in reports.values():
                assert_real(result)
                for candidate in result["candidate_results"]:
                    for attempt in candidate["attempts"]:
                        assert hashlib.sha256(Path(attempt["algorithm_path"]).read_bytes()).hexdigest() == attempt["code_hash"]
            save(output / "extracted_knowledge.json", {"items": workflow.store.list_knowledge_items(1000)})
            shutil.copy2(settings.graphml_path, output / "knowledge_snapshot.graphml")
            source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for folder in ("app", "scripts") for p in (ROOT / folder).rglob("*.py")}
            calls = [call for result in reports.values() for call in result["llm_trace"]["calls"]]
            manifest = {"status": "passed", "timestamp": datetime.now(timezone.utc).isoformat(), "git_head_at_verification": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "source_sha256": source_hashes, "runs": {name: {"run_id": r["run_id"], "task_type": r["spec"]["task_type"], "winner": r["selected_plan"]["algorithm_id"], "metrics": r["validation"]["metrics"]} for name, r in reports.items()}, "llm_calls": len(calls), "providers": sorted({c.get("provider", "unknown") for c in calls}), "models": sorted({str(c.get("model")) for c in calls}), "notes": ["All four workflows asserted real LLM understanding/planning/code/explanation and passing independent validation.", "Separate report extraction calls are retained in report_extraction.json.", "Source hashes snapshot the repository at verification; generated artifact hashes bind every executed algorithm version."]}
            save(output / "manifest.json", manifest)
        print(f"[Acceptance] {current} PASS", flush=True)
    return {"output": str(output), "stages": stages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/acceptance_real_20260907")
    parser.add_argument("--stage", choices=["all", "first", "second", "repair", "cross", "verify"], default="all")
    parser.add_argument("--provider", choices=["openai", "local", "auto"], default="auto")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.output), args.stage, args.provider)))



