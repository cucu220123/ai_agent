from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from app.config import Settings, get_settings
from app.workflow import AlgorithmFactoryWorkflow


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Algorithm Factory CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run end-to-end capability reproduction")
    run.add_argument("--description", required=True)
    run.add_argument("--data", dest="data_path", required=True)
    run.add_argument("--provider", choices=["auto", "mock", "openai", "local"], default=None)
    run.add_argument("--beam-width", type=int, default=None)
    run.add_argument("--max-repairs", type=int, default=None)
    run.add_argument("--cv-folds", type=int, default=0)
    run.add_argument("--source", action="append", type=Path, default=None)
    run.add_argument("--allow-template-fallback", action="store_true", help="explicitly relax real-only acceptance; fallback is labeled")
    sub.add_parser("knowledge", help="list graph knowledge summary")
    sub.add_parser("plugins", help="list registered task and algorithm plugins")
    sub.add_parser("tasks", help="list registered task plugins")
    ingest = sub.add_parser("ingest", help="extract capability knowledge from Markdown/Python")
    ingest.add_argument("path")
    ingest.add_argument("--provider", choices=["auto", "mock", "openai", "local"], default=None)
    args = parser.parse_args()
    if args.command == "run":
        settings = get_settings()
        if args.provider:
            settings = Settings(**{**settings.__dict__, "llm_provider": args.provider})
        settings = replace(settings, beam_width=args.beam_width or settings.beam_width, max_repair_rounds=args.max_repairs if args.max_repairs is not None else settings.max_repair_rounds, validation_cv_folds=args.cv_folds, strict_real_llm=not args.allow_template_fallback)
        if settings.beam_width < 1 or settings.max_repair_rounds < 0 or args.cv_folds < 0:
            parser.error("invalid search/repair/validation budget")
        result = AlgorithmFactoryWorkflow(settings).run(args.description, args.data_path, source_paths=args.source)
        print(json.dumps({"run_id": result.run_id, "status": result.validation.status if result.validation else None, "algorithm": result.selected_plan.algorithm_name if result.selected_plan else None, "metrics": result.validation.metrics if result.validation else {}, "report": result.report_markdown, "generated": result.generated_files}, ensure_ascii=False, indent=2))
        return 0 if result.validation and result.validation.status == "passed" else 1
    if args.command == "knowledge":
        from app.knowledge.store import KnowledgeStore
        settings = get_settings()
        knowledge = KnowledgeStore(settings.knowledge_db, settings.graphml_path)
        print(json.dumps({"summary": knowledge.graph_summary(), "capabilities": knowledge.list_capabilities(), "algorithms": knowledge.list_algorithms(), "recent_runs": knowledge.list_validation_runs(10)}, ensure_ascii=False, indent=2))
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
        settings = get_settings()
        if args.provider:
            settings = Settings(**{**settings.__dict__, "llm_provider": args.provider})
        workflow = AlgorithmFactoryWorkflow(settings)
        print(json.dumps(CapabilityExtractor(workflow.llm, settings.llm_provider).ingest_path(args.path, workflow.store), ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
