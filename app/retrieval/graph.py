from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Callable

import networkx as nx

from app.models import CapabilitySpec


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", value.lower())
    return {w for w in words if len(w) > 1 or w.isascii()}


class SubgraphSerializer:
    def serialize(self, graph: nx.MultiDiGraph, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], spec: CapabilitySpec) -> dict[str, Any]:
        algorithms: list[dict[str, Any]] = []
        node_by_id = {item["id"]: item for item in nodes}
        edge_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        edge_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            edge_by_source[edge["source"]].append(edge)
            edge_by_target[edge["target"]].append(edge)
        for item in nodes:
            if item.get("type") != "Algorithm":
                continue
            incident = edge_by_source.get(item["id"], []) + edge_by_target.get(item["id"], [])
            related = [node_by_id[e["target"] if e["source"] == item["id"] else e["source"]] for e in incident if (e["target"] if e["source"] == item["id"] else e["source"]) in node_by_id]
            algorithms.append({
                "algorithm": item.get("name", item["id"]),
                "algorithm_id": item["id"],
                "applicable_conditions": [n.get("name") for n in related if n.get("type") in {"Task", "Constraint", "FeatureStrategy"}],
                "historical_runs": [n for n in related if n.get("type") == "ValidationRun"],
                "failure_experiences": [n for n in related if n.get("type") == "FailureExperience"],
                "evidence_node_ids": [n["id"] for n in related],
            })
        return {
            "user_requirement": spec.to_dict(),
            "task": {"task_type": spec.task_type, "domain": spec.domain, "target": spec.target_column},
            "candidate_algorithms": algorithms,
            "nodes": nodes,
            "edges": edges,
        }

    def to_text(self, evidence: dict[str, Any]) -> str:
        lines = [f"Task: {evidence['task'].get('task_type')} / {evidence['task'].get('domain')}"]
        for algo in evidence.get("candidate_algorithms", []):
            lines.append(f"Algorithm {algo['algorithm']} ({algo['algorithm_id']})")
            if algo.get("applicable_conditions"):
                lines.append("  conditions: " + ", ".join(str(x) for x in algo["applicable_conditions"]))
            for run in algo.get("historical_runs", []):
                lines.append(f"  validation: status={run.get('status')} metrics={run.get('metrics', {})} dataset={run.get('dataset_id', '')}")
            for failure in algo.get("failure_experiences", []):
                lines.append(f"  failure: {failure.get('summary', failure.get('kind', 'unknown'))}; repair={failure.get('repair_strategy', '')}")
        return "\n".join(lines)


class GraphRetriever:
    """Entity-linking plus bounded multi-hop graph traversal and path scoring."""

    def __init__(self, graph: nx.MultiDiGraph, payload_resolver: Callable[[str], dict[str, Any]] | None = None):
        self.graph = graph
        self.payload_resolver = payload_resolver
        self.serializer = SubgraphSerializer()

    def retrieve(self, spec: CapabilitySpec, hops: int = 2, limit: int = 40) -> dict[str, Any]:
        query_text = " ".join([spec.capability_name, spec.domain, spec.task_type, spec.data_type, spec.target_column, *spec.feature_columns, *spec.metrics, *spec.constraints])
        query_tokens = _tokens(query_text)
        anchors: list[tuple[str, float]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            payload = self.payload_resolver(node_id) if self.payload_resolver else {}
            combined = {**attrs, **payload}
            node_type = attrs.get("type", "")
            text = " ".join(str(v) for v in combined.values())
            node_tokens = _tokens(text + " " + str(node_id))
            overlap = len(query_tokens & node_tokens)
            if combined.get("task_type") == spec.task_type:
                overlap += 5
            if combined.get("domain") == spec.domain:
                overlap += 4
            if spec.target_column and combined.get("target") == spec.target_column:
                overlap += 3
            aliases = combined.get("aliases", [])
            if isinstance(aliases, list) and any(_tokens(str(alias)) & query_tokens for alias in aliases):
                overlap += 3
            if node_type == "Algorithm" and any(node_id.endswith(name) or name.replace("_", " ") in text.lower() for name in spec.candidate_algorithms):
                overlap += 4
            if overlap:
                anchors.append((node_id, float(overlap)))
        anchors = sorted(anchors, key=lambda x: x[1], reverse=True)[:12]
        selected: dict[str, float] = {}
        for anchor, anchor_score in anchors:
            undirected = self.graph.to_undirected()
            lengths = nx.single_source_shortest_path_length(undirected, anchor, cutoff=hops)
            for node_id, distance in lengths.items():
                score = anchor_score / (1.0 + distance)
                selected[node_id] = max(selected.get(node_id, 0.0), score)
        ranked_nodes = sorted(selected.items(), key=lambda x: x[1], reverse=True)[:limit]
        node_ids = {node_id for node_id, _ in ranked_nodes}
        nodes: list[dict[str, Any]] = []
        for node_id, score in ranked_nodes:
            attrs = dict(self.graph.nodes[node_id])
            full_payload = self.payload_resolver(node_id) if self.payload_resolver else {}
            attrs.pop("payload", None)
            nodes.append({"id": node_id, "score": round(score, 5), **attrs, **full_payload})
        edges = []
        for source, target, key, attrs in self.graph.edges(keys=True, data=True):
            if source in node_ids and target in node_ids:
                payload = attrs.get("payload", "")
                try:
                    payload = json.loads(payload) if isinstance(payload, str) and payload else payload
                except json.JSONDecodeError:
                    pass
                edges.append({"source": source, "target": target, "relation": attrs.get("relation", ""), "key": key, "provenance": payload})
        evidence = self.serializer.serialize(self.graph, nodes, edges, spec)
        return {"query": query_text, "anchors": [{"id": x, "score": y} for x, y in anchors], "hops": hops, "nodes": nodes, "edges": edges, "serialized": evidence, "text": self.serializer.to_text(evidence)}
