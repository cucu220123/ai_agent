from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.experience.self_repair import run_self_repair_demo


if __name__ == "__main__":
    result = run_self_repair_demo(Path("examples/self_repair_demo"))
    Path("examples/self_repair_demo/result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"repair_status": result["repair_status"], "first": result["first_validation"]["status"], "second": result["second_validation"]["status"], "diagnosis": result["diagnosis"]}, ensure_ascii=False, indent=2))
