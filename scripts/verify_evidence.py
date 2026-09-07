from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    failures: list[str] = []
    checks: list[tuple[Path, dict]] = []
    for path in sorted((ROOT / "docs/evidence").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path}: invalid JSON {exc}")
            continue
        checks.append((path, payload))
        if payload.get("git_commit"):
            evidence_commit = payload["git_commit"]
            ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", evidence_commit, commit], cwd=ROOT).returncode == 0
            if evidence_commit != commit and not ancestor:
                failures.append(f"{path}: git_commit {evidence_commit} is not an ancestor of HEAD {commit}")
        raw = json.dumps(payload, ensure_ascii=False)
        if re.search(r"sk-[A-Za-z0-9_-]{16,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY", raw):
            failures.append(f"{path}: secret-like content")
    for path in (ROOT / "examples").glob("**/evidence.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("real_llm_success") is True and payload.get("model", "").lower().startswith("mock"):
            failures.append(f"{path}: real_llm_success cannot use mock")
        if payload.get("llm_success") is True and payload.get("generation", {}).get("provider") in {"mock", "template_fallback"}:
            failures.append(f"{path}: llm_success has non-LLM generation source")
    result = {"git_commit": commit, "evidence_files_checked": len(checks), "failures": failures, "status": "passed" if not failures else "failed"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
