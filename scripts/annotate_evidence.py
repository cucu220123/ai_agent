from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for path in list((ROOT / "docs/evidence").glob("*.json")) + list((ROOT / "examples").glob("**/evidence.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload.setdefault("git_commit", commit)
        payload.setdefault("generated_at", generated_at)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(commit)


if __name__ == "__main__":
    main()
