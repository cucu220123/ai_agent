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
    sub.add_parser("knowledge", help="list graph knowledge summary")
    sub.add_parser("plugins", help="list registered task and algorithm plugins")
    sub.add_parser("tasks", help="list registered task plugins")
    ingest = sub.add_parser("ingest", help="extract capability knowledge from Markdown/Python")
    ingest.add_argument("path")
    args = parser.parse_args()
    if args.command == "run":
        settings = get_settings()
        if args.provider:
            settings = Settings(**{**settings.__dict__, "llm_provider": args.provider})
        result = AlgorithmFactoryWorkflow(settings).run(args.description, args.data_path)
        print(json.dumps({"run_id": result.run_id, "status": result.validation.status if result.validation else None, "algorithm": result.selected_plan.algorithm_name if result.selected_plan else None, "metrics": result.validation.metrics if result.validation else {}, "report": result.report_markdown, "generated": result.generated_files}, ensure_ascii=False, indent=2))
        return 0 if result.validation and result.validation.status == "passed" else 1
    if args.command == "knowledge":
        workflow = AlgorithmFactoryWorkflow(get_settings())
        print(json.dumps({"summary": workflow.store.graph_summary(), "capabilities": workflow.store.list_capabilities(), "algorithms": workflow.store.list_algorithms(), "recent_runs": workflow.store.list_validation_runs(10)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "plugins":
        from app.plugins.registry import DEFAULT_REGISTRY

        print(json.dumps(DEFAULT_REGISTRY.describe(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "tasks":
        from app.plugins.registry import DEFAULT_REGISTRY

        print(json.dumps(DEFAULT_REGISTRY.describe()["tasks"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "ingest":
        from app.knowledge.extractor import CapabilityExtractor
        workflow = AlgorithmFactoryWorkflow(get_settings())
        print(json.dumps(CapabilityExtractor().ingest_path(args.path, workflow.store), ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
