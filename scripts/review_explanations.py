"""Review saved explanations without altering measured workflow reports or code."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agents.explanation_agent import ExplanationAgent, validate_comparative_claims
from app.config import get_settings
from app.llm.factory import build_llm
from app.models import CapabilitySpec, KnowledgeContext
from scripts.run_acceptance import save


def review(output: Path, provider: str) -> dict:
    results = {}
    for stage in ("first", "second", "repair", "cross"):
        report = json.loads((output / f"{stage}.json").read_text())
        source_hash = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        error, calls = None, []
        explanation = report["explanation"]
        try:
            validate_comparative_claims(explanation, report["candidate_results"])
        except ValueError as exc:
            error = str(exc)
            llm = build_llm(replace(get_settings(), llm_provider=provider))
            winner = next(c for c in report["candidate_results"] if c["plan"]["algorithm_id"] == report["selected_plan"]["algorithm_id"])
            explanation = ExplanationAgent(llm, provider).run(CapabilitySpec(**report["spec"]), KnowledgeContext(**report["knowledge"]), report["candidate_results"], winner)
            assert explanation["status"] == "ok", explanation
            calls = llm.calls
            assert calls and all("mock" not in str(c["provider"]) for c in calls)
        validate_comparative_claims(explanation, report["candidate_results"])
        result = {"status": "passed", "run_id": report["run_id"], "source_report_semantic_sha256": source_hash, "original_comparison_error": error, "explanation": explanation, "llm_calls": calls, "scope": "Structured numeric/ID checks and supported metric comparison patterns; other free prose still requires review. Original measured report is unchanged."}
        save(output / "explanation_reviews" / f"{report['run_id']}.json", result)
        save(output / "workspace/reports" / f"{report['run_id']}.explanation-review.json", result)
        results[stage] = {"status": "passed", "corrected_with_real_llm": bool(error)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/acceptance_real_20260907"))
    parser.add_argument("--provider", choices=["auto", "openai", "local"], default="auto")
    args = parser.parse_args()
    print(json.dumps(review(args.output, args.provider), indent=2))
