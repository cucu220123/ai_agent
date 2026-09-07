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
        assert json.loads((output / "report_extraction.json").read_text())["extraction_trace"]["status"] == "ok"
        assert (output / "knowledge_snapshot.graphml").is_file()
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


