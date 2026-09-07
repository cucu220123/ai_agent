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
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY, node_type TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_graph(self) -> None:
        with self._connect() as conn:
            for row in conn.execute("SELECT * FROM knowledge_items").fetchall():
                payload = json.loads(row["payload"])
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
        """Materialize metric, environment, dataset and feature strategy entities."""
        capability_id = "cap_churn_prediction_v1"
        task_id = "task_binary_classification"
        self.upsert_knowledge_item(task_id, "Task", {"id": task_id, "name": "binary classification", "task_type": "binary_classification"})
        self.add_edge(capability_id, task_id, "SOLVES")
        for metric_id, name, direction in (("metric_roc_auc", "ROC-AUC", "max"), ("metric_pr_auc", "PR-AUC", "max"), ("metric_f1", "F1", "max"), ("metric_precision", "Precision", "max"), ("metric_recall", "Recall", "max")):
            self.upsert_knowledge_item(metric_id, "Metric", {"id": metric_id, "name": name, "direction": direction})
            self.add_edge(capability_id, metric_id, "EVALUATED_BY")
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
        dataset_id = "dataset_churn_demo"
        self.upsert_knowledge_item(dataset_id, "Dataset", {"id": dataset_id, "path": "data/churn_demo.csv", "target": "churn", "task_type": "binary_classification"})
        self.add_edge(capability_id, dataset_id, "VALIDATED_ON")
        for name in ("numeric_imputation", "categorical_imputation", "one_hot_encoding", "standard_scaling"):
            sid = "feature_" + name
            self.upsert_knowledge_item(sid, "FeatureStrategy", {"id": sid, "name": name})
            self.add_edge(capability_id, sid, "REQUIRES_FEATURE")
        self.export_graph()

    def artifact_fingerprint(self, path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

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

    def upsert_knowledge_item(self, item_id: str, node_type: str, payload: dict[str, Any]) -> None:
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
            self.add_edge(node_id, payload["algorithm_id"], "RELATED_TO")
        self.export_graph()

    def add_algorithm_version(self, item: dict[str, Any]) -> None:
        payload = dict(item)
        node_id = str(payload["id"])
        self.upsert_knowledge_item(node_id, "AlgorithmVersion", payload)
        if payload.get("algorithm_id"):
            self.add_edge(node_id, payload["algorithm_id"], "VERSION_OF")

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
