"""Read-only verification of actual acceptance evidence, portable after clone."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_acceptance import assert_real


def verify(output: Path) -> dict:
    failures = []
    counts = {"runs": 0, "source_versions": 0}
    try:
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["status"] == "passed"
        reports = {}
        for name in ("first", "second", "repair", "cross"):
            report = json.loads((output / f"{name}.json").read_text())
            assert_real(report)
            from app.agents.explanation_agent import validate_comparative_claims
            review_path = output / "explanation_reviews" / f"{report['run_id']}.json"
            checked_explanation = report["explanation"]
            if review_path.exists():
                review = json.loads(review_path.read_text())
                assert review["source_report_semantic_sha256"] == hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
                assert review["status"] == "passed" and review["explanation"]["status"] == "ok"
                checked_explanation = review["explanation"]
                if review["original_comparison_error"]:
                    assert any(c["purpose"] == "explanation" and c["status"] == "ok" and "mock" not in c["provider"] for c in review["llm_calls"])
            validate_comparative_claims(checked_explanation, report["candidate_results"])
            if revision := report.get("report_revision"):
                original_path = output / "report_revisions" / Path(revision["previous_report"]).name
                assert hashlib.sha256(original_path.read_bytes()).hexdigest() == revision["previous_report_sha256"]
                original = json.loads(original_path.read_text())
                for field in ("run_id", "spec", "candidate_results", "selected_plan", "validation", "writeback", "generated_files"):
                    assert original[field] == report[field], f"explanation recovery changed {field}"
            reports[name] = report
            counts["runs"] += 1
            for candidate in report["candidate_results"]:
                for attempt in candidate["attempts"]:
                    relative = attempt["algorithm_path"].split("/workspace/", 1)[1]
                    path = (output / "workspace" / relative).resolve()
                    assert path.is_relative_to((output / "workspace").resolve())
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == attempt["code_hash"]
                    counts["source_versions"] += 1
        first, second = reports["first"], reports["second"]
        assert not first["knowledge"]["historical_cases"]
        assert first["run_id"] in {x["run_id"] for x in second["knowledge"]["historical_cases"]}
        assert first["run_id"] in json.dumps(second["knowledge"]["planning_context"])
        repaired = reports["repair"]["candidate_results"][0]
        assert repaired["attempts"][0]["validation"]["status"] == "failed"
        assert repaired["attempts"][-1]["validation"]["status"] == "passed"
        assert any(x["status"] == "llm_repair_accepted" for x in repaired["repair_history"])
        reused = json.loads((output / "self_repair_demo/next_retrieval.json").read_text())
        assert any(reports["repair"]["run_id"] in str(item) for item in reused["experiences"])
        assert (output / "self_repair_demo/before.py").read_bytes() != (output / "self_repair_demo/after.py").read_bytes()
        assert json.loads((output / "closed_loop_proof.json").read_text())["prior_run_id"] == first["run_id"]
        assert json.loads((output / "report_extraction.json").read_text())["extraction_trace"]["status"] == "ok"
        assert (output / "knowledge_snapshot.graphml").is_file()
        import networkx as nx
        graph = nx.read_graphml(output / "knowledge_snapshot.graphml")
        assert all(report["run_id"] in graph for report in reports.values())
        final_version = repaired["attempts"][-1]["version_id"]
        assert any(target == final_version and attrs.get("relation") == "PRODUCED_VERSION" and graph.nodes[source].get("type") == "RepairExperience" for source, target, attrs in graph.edges(data=True))
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    return {"status": "failed" if failures else "passed", "checked": counts, "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/acceptance_real_20260907"))
    args = parser.parse_args()
    result = verify(args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


