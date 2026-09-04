from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx


class KnowledgeStore:
    """SQLite-backed store with a synchronized NetworkX property graph."""

    def __init__(self, db_path: str | Path, graphml_path: str | Path | None = None):
        self.db_path = Path(db_path)
        self.graphml_path = Path(graphml_path) if graphml_path else self.db_path.with_suffix(".graphml")
        self.graph = nx.MultiDiGraph()
        self._init_db()
        self._load_graph()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS capabilities (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, task_type TEXT,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS algorithms (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS validation_runs (
                    id TEXT PRIMARY KEY, capability_id TEXT, algorithm_id TEXT,
                    status TEXT, payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY, algorithm_id TEXT, kind TEXT,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS graph_edges (
                    source TEXT, target TEXT, relation TEXT, payload TEXT,
                    PRIMARY KEY(source, target, relation)
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_graph(self) -> None:
        with self._connect() as conn:
            for table, node_type in (
                ("capabilities", "Capability"),
                ("algorithms", "Algorithm"),
                ("validation_runs", "ValidationRun"),
                ("experiences", "FailureExperience"),
            ):
                for row in conn.execute(f"SELECT * FROM {table}").fetchall():
                    payload = json.loads(row["payload"])
                    node_id = row["id"]
                    attrs = {k: str(v) for k, v in payload.items() if isinstance(v, (str, int, float, bool))}
                    self.graph.add_node(node_id, type=node_type, **attrs)
            for row in conn.execute("SELECT * FROM graph_edges").fetchall():
                self.graph.add_edge(row["source"], row["target"], relation=row["relation"], payload=row["payload"] or "")

    def seed_from_json(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in data.get("capabilities", []):
            self.upsert_capability(item)
        for item in data.get("algorithms", []):
            self.upsert_algorithm(item)
        for edge in data.get("edges", []):
            self.add_edge(edge["source"], edge["target"], edge["relation"], edge.get("payload", {}))
        self.export_graph()

    def upsert_capability(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["id"])
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO capabilities(id,name,task_type,payload,created_at) VALUES (?,?,?,?,?)",
                (node_id, payload.get("name", node_id), payload.get("task_type", ""), json.dumps(payload, ensure_ascii=False), self._now()),
            )
        self.graph.add_node(node_id, type="Capability", name=payload.get("name", node_id), task_type=payload.get("task_type", ""))

    def upsert_algorithm(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["id"])
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO algorithms(id,name,payload,created_at) VALUES (?,?,?,?)",
                (node_id, payload.get("name", node_id), json.dumps(payload, ensure_ascii=False), self._now()),
            )
        self.graph.add_node(node_id, type="Algorithm", name=payload.get("name", node_id))

    def add_edge(self, source: str, target: str, relation: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        serialized = json.dumps(payload, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO graph_edges(source,target,relation,payload) VALUES (?,?,?,?)",
                (source, target, relation, serialized),
            )
        self.graph.add_edge(source, target, relation=relation, payload=serialized)

    def add_validation_run(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["run_id"])
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO validation_runs(id,capability_id,algorithm_id,status,payload,created_at) VALUES (?,?,?,?,?,?)",
                (node_id, payload.get("capability_id"), payload.get("algorithm_id"), payload.get("status"), json.dumps(payload, ensure_ascii=False), self._now()),
            )
        self.graph.add_node(node_id, type="ValidationRun", status=payload.get("status", ""), algorithm_id=payload.get("algorithm_id", ""))
        if payload.get("capability_id"):
            self.add_edge(node_id, payload["capability_id"], "VALIDATES")
        if payload.get("algorithm_id"):
            self.add_edge(node_id, payload["algorithm_id"], "RELATED_TO")
        self.export_graph()

    def add_experience(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["id"])
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiences(id,algorithm_id,kind,payload,created_at) VALUES (?,?,?,?,?)",
                (node_id, payload.get("algorithm_id"), payload.get("kind", "failure_or_success"), json.dumps(payload, ensure_ascii=False), self._now()),
            )
        self.graph.add_node(node_id, type="FailureExperience", kind=payload.get("kind", ""))
        if payload.get("algorithm_id"):
            self.add_edge(node_id, payload["algorithm_id"], "RELATED_TO")
        self.export_graph()

    def list_capabilities(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [json.loads(row["payload"]) for row in conn.execute("SELECT payload FROM capabilities ORDER BY created_at DESC")]

    def list_algorithms(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [json.loads(row["payload"]) for row in conn.execute("SELECT payload FROM algorithms ORDER BY created_at DESC")]

    def recent_experiences(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [json.loads(row["payload"]) for row in conn.execute("SELECT payload FROM experiences ORDER BY created_at DESC LIMIT ?", (limit,))]

    def list_validation_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [json.loads(row["payload"]) for row in conn.execute("SELECT payload FROM validation_runs ORDER BY created_at DESC LIMIT ?", (limit,))]

    def graph_summary(self) -> dict[str, Any]:
        node_counts: dict[str, int] = {}
        for _, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("type", "Unknown")
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        relation_counts: dict[str, int] = {}
        for _, _, attrs in self.graph.edges(data=True):
            relation = attrs.get("relation", "Unknown")
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges(), "node_types": node_counts, "relations": relation_counts}

    def search(self, query: str, limit: int = 8) -> dict[str, list[dict[str, Any]]]:
        tokens = {t.lower() for t in query.replace("，", " ").replace(",", " ").split() if len(t) > 1}

        def score(item: dict[str, Any]) -> int:
            text = json.dumps(item, ensure_ascii=False).lower()
            return sum(1 for token in tokens if token in text)

        capabilities = sorted(self.list_capabilities(), key=score, reverse=True)[:limit]
        algorithms = sorted(self.list_algorithms(), key=score, reverse=True)[:limit]
        experiences = sorted(self.recent_experiences(limit=limit * 2), key=score, reverse=True)[:limit]
        return {"capabilities": capabilities, "algorithms": algorithms, "experiences": experiences}

    def export_graph(self) -> None:
        self.graphml_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, self.graphml_path)
