from __future__ import annotations

from typing import Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.workflow import AlgorithmFactoryWorkflow
from app.plugins.registry import DEFAULT_REGISTRY


app = FastAPI(title="AI Algorithm Factory", version="0.1.0")


class RunRequest(BaseModel):
    description: str = Field(min_length=5)
    data_path: str
    provider: Literal["auto", "mock", "openai", "local"] = "auto"


class IngestRequest(BaseModel):
    path: str
    provider: Literal["auto", "mock", "openai", "local"] = "auto"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai-algorithm-factory"}


@app.get("/capabilities")
def capabilities() -> dict:
    workflow = AlgorithmFactoryWorkflow()
    return {"items": workflow.store.list_capabilities()}


@app.get("/algorithms")
def algorithms() -> dict:
    workflow = AlgorithmFactoryWorkflow()
    return {"items": workflow.store.list_algorithms()}


@app.get("/plugins")
def plugins() -> dict:
    return DEFAULT_REGISTRY.describe()


@app.get("/tasks")
def tasks() -> dict:
    return {"items": DEFAULT_REGISTRY.describe()["tasks"]}


@app.get("/sources")
def sources(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    workflow = AlgorithmFactoryWorkflow()
    return {"items": workflow.store.list_knowledge_items(limit)}


@app.get("/catalog")
def catalog() -> dict:
    workflow = AlgorithmFactoryWorkflow()
    return {"summary": workflow.store.graph_summary(), "items": workflow.store.list_knowledge_items(200)}


@app.post("/knowledge/ingest")
def ingest_knowledge(request: IngestRequest) -> dict:
    from app.agents.knowledge_extraction_agent import KnowledgeExtractionAgent
    from app.knowledge.extractor import CapabilityExtractor
    from app.config import Settings, get_settings

    base = get_settings()
    requested = Path(request.path)
    resolved = (base.project_root / requested).resolve() if not requested.is_absolute() else requested.resolve()
    if not resolved.exists() or (not resolved.is_file() and not resolved.is_dir()):
        raise HTTPException(status_code=400, detail="source path does not exist")
    if base.project_root.resolve() not in resolved.parents and resolved != base.project_root.resolve():
        raise HTTPException(status_code=400, detail="source path must be inside project directory")
    settings = Settings(**{**base.__dict__, "llm_provider": request.provider})
    workflow = AlgorithmFactoryWorkflow(settings)
    items = CapabilityExtractor(workflow.llm, request.provider).ingest_path(resolved, workflow.store)
    return {"count": len(items), "items": items, "graph_summary": workflow.store.graph_summary()}


@app.get("/run/{run_id}")
def run_detail(run_id: str) -> dict:
    workflow = AlgorithmFactoryWorkflow()
    for item in workflow.store.list_validation_runs(200):
        if item.get("run_id") == run_id:
            return item
    raise HTTPException(status_code=404, detail="run not found")


@app.get("/knowledge/search")
def knowledge_search(q: str = Query(min_length=1), limit: int = Query(default=8, ge=1, le=50)) -> dict:
    workflow = AlgorithmFactoryWorkflow()
    return workflow.store.search(q, limit=limit)


@app.get("/graph/summary")
def graph_summary() -> dict:
    workflow = AlgorithmFactoryWorkflow()
    return workflow.store.graph_summary()


@app.get("/runs")
def runs(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    workflow = AlgorithmFactoryWorkflow()
    return {"items": workflow.store.list_validation_runs(limit)}


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>AI Algorithm Factory</title>
    <style>body{font-family:system-ui;margin:40px;max-width:960px}textarea,input{width:100%;padding:10px;margin:6px 0 14px}button{padding:10px 18px;background:#1f6feb;color:white;border:0;border-radius:5px}pre{background:#f6f8fa;padding:16px;overflow:auto}</style></head>
    <body><h1>AI Algorithm Factory</h1><p>提交算法能力描述，系统将自动解析、检索、生成、验证并沉淀。</p>
    <label>能力描述</label><textarea id='description' rows='5'>根据客户年龄、地区、登录频率、消费金额和投诉次数预测客户是否流失，要求 ROC-AUC 不低于 0.75，并输出概率。</textarea>
    <label>数据路径</label><input id='data_path' value='data/churn_demo.csv'/><button onclick='run()'>运行工作流</button><h2>结果</h2><pre id='result'>等待运行...</pre>
    <script>async function run(){const result=document.getElementById('result');result.textContent='运行中...';const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({description:document.getElementById('description').value,data_path:document.getElementById('data_path').value,provider:'mock'})});result.textContent=JSON.stringify(await r.json(),null,2)}</script></body></html>"""


@app.post("/run")
def run(request: RunRequest) -> dict:
    try:
        from app.config import Settings, get_settings

        base = get_settings()
        requested = Path(request.data_path)
        resolved = (base.project_root / requested).resolve() if not requested.is_absolute() else requested.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"data file does not exist: {resolved}")
        project_root = base.project_root.resolve()
        if project_root not in resolved.parents and resolved != project_root:
            raise ValueError("API data_path must be inside the project directory")
        settings = Settings(**{**base.__dict__, "llm_provider": request.provider})
        result = AlgorithmFactoryWorkflow(settings).run(request.description, resolved)
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"workflow failed: {type(exc).__name__}: {exc}") from exc
