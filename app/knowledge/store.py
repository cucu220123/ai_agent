from __future__ import annotations

import json
import hashlib
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
        self._payloads: dict[str, dict[str, Any]] = {}
        self._init_db()
        self._load_graph()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
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
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY, node_type TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_graph(self) -> None:
        self.graph.clear()
        self._payloads.clear()
        with self._connect() as conn:
            for row in conn.execute("SELECT * FROM knowledge_items").fetchall():
                payload = json.loads(row["payload"])
                self._payloads[row["id"]] = payload
                attrs = {k: str(v) for k, v in payload.items() if isinstance(v, (str, int, float, bool))}
                self.graph.add_node(row["id"], type=row["node_type"], **attrs)
            for table, node_type in (
                ("capabilities", "Capability"),
                ("algorithms", "Algorithm"),
                ("validation_runs", "ValidationRun"),
                ("experiences", "FailureExperience"),
            ):
                for row in conn.execute(f"SELECT * FROM {table}").fetchall():
                    payload = json.loads(row["payload"])
                    self._payloads[row["id"]] = payload
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

    def ensure_catalog_nodes(self) -> None:
        """Materialize only generic catalog entities; domain data comes from seed/extraction/ingestion."""
        from app.metrics.registry import METRIC_REGISTRY
        from app.plugins.registry import DEFAULT_REGISTRY

        for plugin in DEFAULT_REGISTRY.algorithms.values():
            algorithm_id = f"algorithm_{plugin.id}"
            if algorithm_id not in self._payloads:
                self.upsert_algorithm({"id": algorithm_id, "name": plugin.name, "task_types": plugin.task_types, "resource_profile": plugin.resource_profile, "preprocessing": plugin.preprocessing, "origin": "plugin_catalog"})

        for task_id, task in DEFAULT_REGISTRY.tasks.items():
            node_id = f"task_{task_id}"
            self.upsert_knowledge_item(node_id, "Task", {"id": node_id, "name": task.name, "task_type": task_id, "target_kind": task.target_kind})
            for metric_name in task.default_metrics:
                definition = METRIC_REGISTRY.metrics.get(metric_name)
                metric_id = f"metric_{metric_name}"
                self.upsert_knowledge_item(metric_id, "Metric", {"id": metric_id, "name": metric_name, "direction": "max" if not definition or definition.maximize else "min", "task_types": list(definition.tasks) if definition else [task_id]})
                self.add_edge(node_id, metric_id, "EVALUATED_BY")
        env_id = "environment_python_sklearn"
        self.upsert_knowledge_item(env_id, "Environment", {"id": env_id, "python": "3.10+", "dependencies": ["pandas", "numpy", "scikit-learn"]})
        for algorithm in self.list_algorithms():
            self.add_edge(algorithm["id"], env_id, "REQUIRES")
            for task_type in algorithm.get("task_types", []):
                task_node = f"task_{task_type}"
                self.upsert_knowledge_item(task_node, "Task", {"id": task_node, "name": task_type, "task_type": task_type})
                self.add_edge(algorithm["id"], task_node, "SUITABLE_FOR")
            dependency_id = "dependency_scikit_learn"
            self.upsert_knowledge_item(dependency_id, "Dependency", {"id": dependency_id, "name": "scikit-learn", "version": ">=1.2"})
            self.add_edge(algorithm["id"], dependency_id, "REQUIRES")
        self.export_graph()

    def artifact_fingerprint(self, path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def upsert_capability(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["id"])
        self._payloads[node_id] = payload
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO capabilities(id,name,task_type,payload,created_at) VALUES (?,?,?,?,?)",
                (node_id, payload.get("name", node_id), payload.get("task_type", ""), json.dumps(payload, ensure_ascii=False), self._now()),
            )
        self.graph.add_node(node_id, type="Capability", name=payload.get("name", node_id), task_type=payload.get("task_type", ""))

    def upsert_algorithm(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["id"])
        self._payloads[node_id] = payload
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO algorithms(id,name,payload,created_at) VALUES (?,?,?,?)",
                (node_id, payload.get("name", node_id), json.dumps(payload, ensure_ascii=False), self._now()),
            )
        self.graph.add_node(node_id, type="Algorithm", name=payload.get("name", node_id))

    def upsert_knowledge_item(self, item_id: str, node_type: str, payload: dict[str, Any]) -> None:
        payload = {**payload, "id": item_id}
        self._payloads[item_id] = payload
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO knowledge_items(id,node_type,payload,created_at) VALUES (?,?,?,?)", (item_id, node_type, json.dumps(payload, ensure_ascii=False), self._now()))
        attrs = {k: str(v) for k, v in payload.items() if isinstance(v, (str, int, float, bool))}
        self.graph.add_node(item_id, type=node_type, **attrs)

    def list_knowledge_items(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [json.loads(row["payload"]) for row in conn.execute("SELECT payload FROM knowledge_items ORDER BY created_at DESC LIMIT ?", (limit,))]

    def add_edge(self, source: str, target: str, relation: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        serialized = json.dumps(payload, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO graph_edges(source,target,relation,payload) VALUES (?,?,?,?)",
                (source, target, relation, serialized),
            )
        for key, attrs in list(self.graph.get_edge_data(source, target, default={}).items()):
            if attrs.get("relation") == relation:
                self.graph.remove_edge(source, target, key=key)
        self.graph.add_edge(source, target, relation=relation, payload=serialized)

    def add_validation_run(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["run_id"])
        self._payloads[node_id] = payload
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO validation_runs(id,capability_id,algorithm_id,status,payload,created_at) VALUES (?,?,?,?,?,?)",
                (node_id, payload.get("capability_id"), payload.get("algorithm_id"), payload.get("status"), json.dumps(payload, ensure_ascii=False), self._now()),
            )
        metric_attrs = {f"metric_{k}": str(v) for k, v in payload.get("metrics", {}).items() if isinstance(v, (int, float))}
        self.graph.add_node(node_id, type="ValidationRun", status=payload.get("status", ""), algorithm_id=payload.get("algorithm_id", ""), task_type=payload.get("task_type", ""), timestamp=payload.get("timestamp", ""), **metric_attrs)
        if payload.get("dataset_id"):
            self.add_edge(node_id, payload["dataset_id"], "ON_DATASET")
        if payload.get("config_id"):
            self.add_edge(node_id, payload["config_id"], "HAS_CONFIG")
        if payload.get("capability_id"):
            self.add_edge(node_id, payload["capability_id"], "VALIDATES")
        if payload.get("algorithm_id"):
            self.add_edge(node_id, payload["algorithm_id"], "VALIDATES")
        if payload.get("version_id"):
            self.add_edge(node_id, payload["version_id"], "PRODUCED_VERSION")
        self.export_graph()

    def add_algorithm_version(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["id"])
        self.upsert_knowledge_item(node_id, "AlgorithmVersion", payload)
        if payload.get("algorithm_id"):
            self.add_edge(node_id, payload["algorithm_id"], "VERSION_OF")
        if payload.get("parent_version"):
            self.add_edge(node_id, payload["parent_version"], "PARENT_VERSION")

    def add_repair_experience(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["id"])
        self.upsert_knowledge_item(node_id, "RepairExperience", payload)
        if payload.get("failure_id"):
            self.add_edge(node_id, payload["failure_id"], "REPAIRS")

    def add_source_support(self, source_id: str, target_id: str, relation: str = "SUPPORTS") -> None:
        self.add_edge(source_id, target_id, relation)

    def add_experience(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["id"])
        self._payloads[node_id] = payload
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiences(id,algorithm_id,kind,payload,created_at) VALUES (?,?,?,?,?)",
                (node_id, payload.get("algorithm_id"), payload.get("kind", "failure_or_success"), json.dumps(payload, ensure_ascii=False), self._now()),
            )
        self.graph.add_node(node_id, type="FailureExperience", kind=payload.get("kind", ""), failure_type=payload.get("failure_type", ""), root_cause=payload.get("root_cause", ""), summary=payload.get("summary", ""))
        if payload.get("algorithm_id"):
            self.add_edge(node_id, payload["algorithm_id"], "RELATED_TO")
        if payload.get("run_id"):
            self.add_edge(node_id, payload["run_id"], "OCCURRED_IN")
        if payload.get("repair_experience_id"):
            self.add_edge(payload["repair_experience_id"], node_id, "REPAIRS")
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

    def get_node_payload(self, node_id: str) -> dict[str, Any]:
        """Resolve full JSON payload for GraphRAG serialization, not lossy GraphML scalars."""
        if node_id in self._payloads:
            return dict(self._payloads[node_id])
        table_specs = (("capabilities", "id"), ("algorithms", "id"), ("validation_runs", "id"), ("experiences", "id"), ("knowledge_items", "id"))
        with self._connect() as conn:
            for table, column in table_specs:
                row = conn.execute(f"SELECT payload FROM {table} WHERE {column} = ?", (node_id,)).fetchone()
                if row:
                    return json.loads(row["payload"])
        return dict(self.graph.nodes[node_id]) if node_id in self.graph else {}

    def refresh(self) -> None:
        """Reload persisted mutations while retaining the graph object's identity."""
        self._load_graph()

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
