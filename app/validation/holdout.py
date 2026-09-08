"""One final evaluation after selection, with an immutable code/data commitment.

This is a reproducibility guard for a cooperative local workflow, not a security
boundary against an operator deleting the ledger or creating another directory.
No final-test prediction, label or score is returned to a repair/planning agent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from dataclasses import replace
from datetime import datetime, timezone

from app.models import CapabilitySpec
from app.validation.runner import ValidationRunner


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalHoldoutEvaluator:
    def __init__(self, validator: ValidationRunner | None = None):
        self.validator = validator or ValidationRunner()

    def evaluate(self, selection: dict[str, Any], development: Path, final_test: Path, output: Path) -> dict[str, Any]:
        if selection["validation"]["status"] != "passed":
            raise ValueError("final evaluation requires a passing development winner")
        winner_id = selection["selected_plan"]["algorithm_id"]
        winner = next(c for c in selection["candidate_results"] if c["plan"]["algorithm_id"] == winner_id)
        source = Path(winner["algorithm_path"])
        source_hash = sha256(source)
        if source_hash != winner["artifact_sha256"]:
            raise ValueError("winner source changed after development selection")
        dev_hash = sha256(development)
        if selection["spec"]["dataset_profile"]["sha256"] != dev_hash:
            raise ValueError("development data changed after selection")
        commitment = {"selection_run_id": selection["run_id"], "selection_report_sha256": hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(), "winner_id": winner_id, "source_sha256": source_hash, "development_sha256": dev_hash, "final_test_sha256": sha256(final_test)}
        output.mkdir(parents=True, exist_ok=True)
        lock, result_path = output / "commitment.json", output / "result.json"
        if lock.exists():
            previous = json.loads(lock.read_text())
            if previous["commitment"] != commitment:
                raise ValueError("final evaluation already committed to different code/data/selection")
            if result_path.exists():
                return json.loads(result_path.read_text())
            raise RuntimeError("final evaluation was interrupted; ledger retained, automatic re-evaluation prohibited")
        # Exclusive creation prevents concurrent consumers of the same final test.
        with lock.open("x", encoding="utf-8") as handle:
            json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "commitment": commitment}, handle, indent=2)
        frozen = output / "frozen_winner.py"
        frozen.write_bytes(source.read_bytes())
        if sha256(frozen) != source_hash:
            raise ValueError("source changed while freezing winner")
        spec = replace(CapabilitySpec(**selection["spec"]), dataset_path=str(development))
        validation = self.validator.run(frozen, development, spec, winner["plan"]["algorithm_name"], config=winner["plan"]["hyperparameters"], evaluation_data_path=final_test)
        if sha256(development) != dev_hash or sha256(final_test) != commitment["final_test_sha256"]:
            raise ValueError("data changed during final evaluation")
        result = {"status": validation.status, "timestamp": datetime.now(timezone.utc).isoformat(), "commitment": commitment, "validation": validation.to_dict(), "selection_policy": "winner fixed by development results before reading final metrics; no repair or candidate substitution after final evaluation", "final_test_feedback_to_agents": False, "cached_rerun_only": True}
        temporary = result_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(result_path)
        return result
