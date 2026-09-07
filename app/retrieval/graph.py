"""Relation-aware, bounded graph search with explicit reversible path evidence."""
from __future__ import annotations

import json
import re
from collections import deque
from typing import Any, Callable

import networkx as nx
from app.models import CapabilitySpec


def _tokens(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_]+", value.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", value):
        tokens.update(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return tokens


RELATION_WEIGHTS = {
    "SOLVES": 1.0, "USES_ALGORITHM": 1.0, "SUITABLE_FOR": 1.0,
    "VALIDATES": 0.95, "ON_DATASET": 0.85, "HAS_CONFIG": 0.8,
    "OCCURRED_IN": 0.95, "REPAIRS": 1.0, "SUPPORTS": 0.8,
    "USES_PREPROCESSING": 0.85, "EVALUATED_BY": 0.7,
    "REQUIRES": 0.6, "REQUIRES_FEATURE": 0.6, "RELATED_TO": 0.75,
    "SATISFIES": 0.8, "HAS_INPUT": 0.7, "OUTPUTS": 0.7,
    "PREDICTS": 0.7, "ACCEPTS": 0.7, "VERSION_OF": 0.6,
    "PRODUCED_VERSION": 0.6, "PARENT_VERSION": 0.5, "VALIDATED_ON": 0.85,
}
TERMINAL_TYPES = {"Dependency", "Environment", "Metric", "Feature", "Target", "InputSchema", "OutputSchema", "AlgorithmVersion", "HyperparameterConfig"}


class SubgraphSerializer:
    """Serialize actual graph paths; never ask an LLM to interpret GraphML bytes."""
    def serialize(self, graph: nx.MultiDiGraph, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], spec: CapabilitySpec) -> dict[str, Any]:
        node_by_id = {item["id"]: item for item in nodes}
        adjacency: dict[str, set[str]] = {key: set() for key in node_by_id}
        for edge in edges:
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])
        algorithms = []
        for item in nodes:
            if item.get("type") != "Algorithm":
                continue
            direct = adjacency[item["id"]]
            # Only follow this algorithm's runs/failures to avoid sibling contamination.
            run_ids = {key for key in direct if node_by_id[key].get("type") == "ValidationRun"}
            failure_ids = {key for key in direct if node_by_id[key].get("type") == "FailureExperience"}
            failure_ids |= {key for run in run_ids for key in adjacency[run] if node_by_id[key].get("type") == "FailureExperience"}
            repair_ids = {key for failure in failure_ids for key in adjacency[failure] if node_by_id[key].get("type") == "RepairExperience"}
            evidence_ids = direct | run_ids | failure_ids | repair_ids
            algorithms.append({
                "algorithm": item.get("name", item["id"]), "algorithm_id": item["id"],
                "applicable_conditions": [node_by_id[key].get("name", key) for key in sorted(direct) if node_by_id[key].get("type") in {"Task", "Constraint", "FeatureStrategy", "PreprocessingStrategy"}],
                "historical_runs": [node_by_id[key] for key in sorted(run_ids)],
                "failure_experiences": [node_by_id[key] for key in sorted(failure_ids)],
                "repair_experiences": [node_by_id[key] for key in sorted(repair_ids)],
                "evidence_node_ids": sorted(evidence_ids),
            })
        return {"task": {"task_type": spec.task_type, "domain": spec.domain, "target": spec.target_column}, "candidate_algorithms": algorithms, "nodes": nodes, "edges": edges}

    def to_text(self, evidence: dict[str, Any]) -> str:
        return json.dumps({"task": evidence["task"], "candidate_algorithms": evidence["candidate_algorithms"]}, ensure_ascii=False)


class GraphRetriever:
    def __init__(self, graph: nx.MultiDiGraph, payload_resolver: Callable[[str], dict[str, Any]] | None = None):
        self.graph = graph
        self.payload_resolver = payload_resolver
        self.serializer = SubgraphSerializer()

    def retrieve(self, spec: CapabilitySpec, hops: int = 3, limit: int = 64) -> dict[str, Any]:
        if not 1 <= hops <= 3:
            raise ValueError("graph hop budget must be 1..3")
        payloads = {node: {**attrs, **(self.payload_resolver(node) if self.payload_resolver else {})} for node, attrs in self.graph.nodes(data=True)}
        query = " ".join([spec.raw_description, spec.domain, spec.capability_name, spec.task_type])
        query_tokens = _tokens(query)
        anchors = []
        for node, data in payloads.items():
            kind = self.graph.nodes[node].get("type")
            if kind not in {"Capability", "Task", "Algorithm"} or not self._compatible(data, spec):
                continue
            lexical = len(query_tokens & _tokens(" ".join(str(data.get(k, "")) for k in ("name", "domain", "aliases"))))
            exact_task = data.get("task_type") == spec.task_type
            exact_domain = data.get("domain") == spec.domain
            score = min(lexical, 5) + 4 * exact_task + 5 * exact_domain
            reasons = [name for name, matched in (("task_type", exact_task), ("domain", exact_domain), ("name_alias", lexical > 0)) if matched]
            if kind == "Algorithm":
                explicit = any(h.lower().replace(" ", "_") in {node.removeprefix("algorithm_"), str(data.get("name", "")).lower().replace(" ", "_")} for h in spec.candidate_hints)
                if not explicit:
                    continue
                score += 5
                reasons.append("explicit_algorithm_hint")
            if score:
                anchors.append({"id": node, "score": float(score), "reasons": reasons, "type": kind})
        anchors.sort(key=lambda x: (-x["score"], x["id"]))
        anchors = anchors[:6]
        if not anchors:
            anchors = [{"id": node, "score": 1.0, "type": "Algorithm", "reasons": ["compatible_catalog_fallback"]} for node, data in payloads.items() if self.graph.nodes[node].get("type") == "Algorithm" and spec.task_type in data.get("task_types", [])][:6]
        best: dict[str, dict[str, Any]] = {}
        for anchor in anchors:
            queue = deque([(anchor["id"], [anchor["id"]], [], anchor["score"])])
            seen = {anchor["id"]: 0}
            while queue:
                node, path, relations, score = queue.popleft()
                distance = len(path) - 1
                if node not in best or score > best[node]["score"]:
                    best[node] = {"score": score, "distance": distance, "anchor": anchor["id"], "node_path": path, "edge_path": relations}
                kind = self.graph.nodes[node].get("type")
                if distance >= hops or (distance and (kind in TERMINAL_TYPES or kind == "Task")):
                    continue
                incident = [(u, v, a, "out") for u, v, a in self.graph.out_edges(node, data=True)] + [(u, v, a, "in") for u, v, a in self.graph.in_edges(node, data=True)]
                for u, v, attrs, direction in incident:
                    relation = attrs.get("relation")
                    neighbor = v if direction == "out" else u
                    if relation not in RELATION_WEIGHTS or neighbor in path or not self._compatible(payloads[neighbor], spec):
                        continue
                    if seen.get(neighbor, hops + 1) <= distance + 1:
                        continue
                    seen[neighbor] = distance + 1
                    edge = {"source": u, "target": v, "relation": relation, "direction": direction}
                    queue.append((neighbor, path + [neighbor], relations + [edge], score * 0.72 * RELATION_WEIGHTS[relation]))
        ranked = sorted(best, key=lambda n: (-best[n]["score"], best[n]["distance"], n))
        # Preserve intermediate path nodes even when the evidence budget truncates leaves.
        selected: set[str] = set()
        for node in ranked:
            addition = set(best[node]["node_path"]) - selected
            if len(selected) + len(addition) <= limit:
                selected.update(addition)
        nodes = [{**payloads[node], "id": node, "type": self.graph.nodes[node].get("type"), "score": round(best[node]["score"], 6), "graph_distance": best[node]["distance"]} for node in ranked if node in selected]
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            if u in selected and v in selected and attrs.get("relation") in RELATION_WEIGHTS:
                provenance = attrs.get("payload", {})
                if isinstance(provenance, str):
                    try: provenance = json.loads(provenance)
                    except ValueError: provenance = {}
                edges.append({"source": u, "target": v, "relation": attrs["relation"], "provenance": provenance})
        serialized = self.serializer.serialize(self.graph, nodes, edges, spec)
        return {"query": query, "anchors": anchors, "hops": hops, "nodes": nodes, "edges": edges, "paths": [{"target": node, **best[node]} for node in ranked if node in selected], "serialized": serialized, "text": self.serializer.to_text(serialized)}

    @staticmethod
    def _compatible(data: dict[str, Any], spec: CapabilitySpec) -> bool:
        task = data.get("task_type")
        tasks = data.get("task_types", [])
        if task and task != spec.task_type:
            return False
        if isinstance(tasks, list) and tasks and spec.task_type not in tasks:
            return False
        return True

