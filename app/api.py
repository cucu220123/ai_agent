"""Local research API. Read-only views never instantiate an LLM."""
from __future__ import annotations
import json
import hashlib
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Literal
import networkx as nx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from app.config import get_settings
from app.knowledge.store import KnowledgeStore
from app.llm.security import sanitize
from app.plugins.registry import DEFAULT_REGISTRY
from app.workflow import AlgorithmFactoryWorkflow

app = FastAPI(title="AI Algorithm Factory", version="0.2.0")
mutation_lock = threading.Lock()


class RunRequest(BaseModel):
    description: str = Field(min_length=5, max_length=12000)
    data_path: str
    provider: Literal["auto", "mock", "openai", "local"] = "auto"
    beam_width: int = Field(default=3, ge=1, le=6)
    max_repair_rounds: int = Field(default=3, ge=0, le=5)


class IngestRequest(BaseModel):
    path: str
    provider: Literal["auto", "mock", "openai", "local"] = "auto"


def store() -> KnowledgeStore:
    settings = get_settings()
    return KnowledgeStore(settings.knowledge_db, settings.graphml_path)


def project_path(value: str) -> Path:
    root = get_settings().project_root.resolve()
    requested = Path(value)
    resolved = (root / requested).resolve()
    if not resolved.is_relative_to(root) or not resolved.exists():
        raise HTTPException(400, "path must exist inside the project workspace")
    if any(part.lower().startswith(("secret", ".env", ".git")) for part in resolved.relative_to(root).parts):
        raise HTTPException(400, "credential and Git paths are not accessible")
    return resolved


def report_path(run_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", run_id):
        raise HTTPException(404, "run not found")
    path = get_settings().reports_dir / f"{run_id}.json"
    if not path.is_file():
        raise HTTPException(404, "run report not found")
    return path


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai-algorithm-factory"}


@app.get("/capabilities")
def capabilities() -> dict:
    return {"items": store().list_capabilities()}


@app.get("/algorithms")
def algorithms() -> dict:
    return {"items": store().list_algorithms()}


@app.get("/plugins")
def plugins() -> dict:
    return DEFAULT_REGISTRY.describe()


@app.get("/tasks")
def tasks() -> dict:
    return {"items": DEFAULT_REGISTRY.describe()["tasks"]}


@app.get("/sources")
def sources(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    return {"items": store().list_knowledge_items(limit)}


@app.get("/catalog")
def catalog() -> dict:
    knowledge = store()
    return {"summary": knowledge.graph_summary(), "items": knowledge.list_knowledge_items(200)}


@app.post("/knowledge/ingest")
def ingest_knowledge(request: IngestRequest) -> dict:
    from app.knowledge.extractor import CapabilityExtractor
    from app.llm.factory import build_llm
    resolved = project_path(request.path)
    settings = replace(get_settings(), llm_provider=request.provider)
    try:
        with mutation_lock:
            knowledge = store()
            items = CapabilityExtractor(build_llm(settings), request.provider).ingest_path(resolved, knowledge)
            return sanitize({"count": len(items), "items": items, "graph_summary": knowledge.graph_summary()})
    except Exception as exc:
        raise HTTPException(400, sanitize(f"{type(exc).__name__}: {exc}")) from exc


@app.get("/run/{run_id}")
def run_detail(run_id: str) -> dict:
    report = json.loads(report_path(run_id).read_text())
    original_digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    review_path = get_settings().reports_dir / f"{run_id}.explanation-review.json"
    if review_path.is_file():
        review = json.loads(review_path.read_text())
        if review.get("status") == "passed" and review.get("source_report_semantic_sha256") == original_digest:
            report["explanation_review"] = review
    final_path = get_settings().reports_dir / f"{run_id}.final-evaluation.json"
    if final_path.is_file():
        final = json.loads(final_path.read_text())
        if final.get("commitment", {}).get("selection_report_sha256") == original_digest and final["commitment"].get("selection_run_id") == run_id:
            report["final_holdout"] = final
    return sanitize(report)


@app.get("/run/{run_id}/code")
def generated_code(run_id: str, candidate: str = Query(min_length=1)) -> dict:
    report = run_detail(run_id)
    item = next((c for c in report["candidate_results"] if c["plan"]["algorithm_id"] == candidate), None)
    if item is None:
        raise HTTPException(404, "candidate not found")
    path = Path(item["algorithm_path"]).resolve()
    root = get_settings().generated_dir.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "generated artifact unavailable")
    return {"source": path.read_text(), "sha256": item["artifact_sha256"]}


@app.get("/knowledge/search")
def knowledge_search(q: str = Query(min_length=1), limit: int = Query(default=8, ge=1, le=50)) -> dict:
    # Legacy lexical inspector; /run uses typed GraphRAG.
    return store().search(q, limit=limit)


@app.get("/graph/summary")
def graph_summary() -> dict:
    return store().graph_summary()


@app.get("/graph/subgraph")
def graph_subgraph(focus: str, hops: int = Query(default=2, ge=1, le=3)) -> dict:
    knowledge = store()
    graph = knowledge.graph
    anchors = [n for n in graph if n == focus or knowledge.get_node_payload(n).get("workflow_run_id") == focus]
    if not anchors:
        raise HTTPException(404, "graph focus not found")
    selected = set()
    for anchor in anchors[:10]:
        selected.update(nx.single_source_shortest_path_length(graph.to_undirected(), anchor, cutoff=hops))
    selected = set(sorted(selected)[:120])
    return {"nodes": [{**knowledge.get_node_payload(n), "id": n, "type": graph.nodes[n].get("type")} for n in selected], "edges": [{"source": u, "target": v, "relation": a.get("relation")} for u, v, a in graph.edges(data=True) if u in selected and v in selected], "focus": focus, "hops": hops}


@app.get("/runs")
def runs(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    return {"items": store().list_validation_runs(limit)}


@app.get("/reports")
def reports() -> dict:
    items = []
    for path in sorted(get_settings().reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
        if not re.fullmatch(r"[a-f0-9]{12}", path.stem):
            continue
        record = json.loads(path.read_text())
        items.append({"run_id": record["run_id"], "status": (record.get("validation") or {}).get("status"), "task_type": record["spec"]["task_type"]})
    return {"items": items}


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return Path(__file__).with_name("ui").joinpath("index.html").read_text(encoding="utf-8-sig")


@app.post("/run")
def run(request: RunRequest) -> dict:
    resolved = project_path(request.data_path)
    if not resolved.is_file():
        raise HTTPException(400, "dataset must be a file")
    settings = replace(get_settings(), llm_provider=request.provider, beam_width=request.beam_width, max_repair_rounds=request.max_repair_rounds)
    try:
        with mutation_lock:
            result = AlgorithmFactoryWorkflow(settings).run(request.description, resolved)
        return sanitize(result.to_dict())
    except Exception as exc:
        raise HTTPException(400, sanitize(f"workflow failed: {type(exc).__name__}: {exc}")) from exc

