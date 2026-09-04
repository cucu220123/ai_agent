from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Settings, get_settings
from app.workflow import AlgorithmFactoryWorkflow


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Algorithm Factory CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run end-to-end capability reproduction")
    run.add_argument("--description", required=True)
    run.add_argument("--data", dest="data_path", required=True)
    run.add_argument("--provider", choices=["mock", "openai", "local"], default=None)
    args = parser.parse_args()
    if args.command == "run":
        settings = get_settings()
        if args.provider:
            settings = Settings(**{**settings.__dict__, "llm_provider": args.provider})
        result = AlgorithmFactoryWorkflow(settings).run(args.description, args.data_path)
        print(json.dumps({"run_id": result.run_id, "status": result.validation.status if result.validation else None, "algorithm": result.selected_plan.algorithm_name if result.selected_plan else None, "metrics": result.validation.metrics if result.validation else {}, "report": result.report_markdown, "generated": result.generated_files}, ensure_ascii=False, indent=2))
        return 0 if result.validation and result.validation.status == "passed" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
