"""Typed workflow messages and coordinator-enforced specialist tool capabilities."""
from __future__ import annotations
from datetime import datetime, timezone
import os
import json
from typing import Any, Callable
from pydantic import BaseModel, Field
from app.llm.security import sanitize

TOOLS = {
    "RequirementAgent": {"understand"}, "KnowledgeExtractionAgent": {"ingest"},
    "RetrievalAgent": {"retrieve"}, "PlannerAgent": {"plan", "search", "advise"},
    "CoderAgent": {"generate"}, "ValidatorAgent": {"validate"},
    "CriticAgent": {"critique"}, "RepairAgent": {"repair"},
    "CuratorAgent": {"curate"}, "ExplanationAgent": {"explain"},
}


class AgentEvent(BaseModel):
    sequence: int = Field(ge=1)
    timestamp: str
    agent: str
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class AgentRuntime:
    def __init__(self, events: list[dict[str, Any]]):
        self.events = events

    def emit(self, agent: str, status: str, **details: Any) -> None:
        event = AgentEvent(sequence=len(self.events) + 1, timestamp=datetime.now(timezone.utc).isoformat(), agent=agent, status=status, details=sanitize(details))
        # Keep top-level details for compatibility with existing report/UI consumers.
        self.events.append({**event.model_dump(exclude={"details"}), **event.details})
        if os.getenv("AI_FACTORY_TRACE", "0") == "1":
            concise = {k: v for k, v in event.details.items() if k in {"tool", "algorithm", "round", "metrics", "version_id"}}
            print(f"[{agent}] {status} " + json.dumps(concise, ensure_ascii=False), flush=True)

    def call(self, agent: str, tool: str, function: Callable, *args: Any, **kwargs: Any) -> Any:
        if tool not in TOOLS.get(agent, set()):
            raise PermissionError(f"{agent} is not allowed to invoke {tool}")
        self.emit(agent, "started", tool=tool)
        try:
            output = function(*args, **kwargs)
        except Exception as exc:
            self.emit(agent, "error", tool=tool, error=str(exc))
            raise
        self.emit(agent, "completed", tool=tool)
        return output

