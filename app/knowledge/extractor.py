from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.agents.knowledge_extraction_agent import KnowledgeExtractionAgent
from app.knowledge.store import KnowledgeStore


class CapabilityExtractor:
    """Full-source chunking + AST facts + strict LLM semantic extraction."""

    def __init__(self, llm=None, provider_name: str = "mock", chunk_chars: int = 5000):
        self.semantic_agent = KnowledgeExtractionAgent(llm, provider_name)
        self.chunk_chars = chunk_chars

    def extract_markdown(self, path: str | Path) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        headings = re.findall(r"^(#+)\s+(.+)$", text, flags=re.MULTILINE)
        metrics = sorted(set(re.findall(r"\b(?:ROC-AUC|PR-AUC|F1|Precision|Recall|RMSE|MAE|R2)\b", text, flags=re.I)))
        fields = re.findall(r"^\|\s*([A-Za-z_][\w]*)\s*\|", text, flags=re.MULTILINE)
        return {
            "source": str(path), "kind": "markdown", "headings": [name for _, name in headings],
            "heading_hierarchy": [{"level": len(markers), "title": title} for markers, title in headings],
            "metrics": metrics, "fields": fields, "character_count": len(text),
            "summary": text[:1000], "content_chunks": self._chunk_document(text),
        }

    def extract_python(self, path: str | Path) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(text)
        functions, classes, imports, relevant_chunks = [], [], [], []
        relevance = re.compile(r"train|predict|evaluate|preprocess|pipeline|metric|model|estimator|transform", re.I)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                source = ast.get_source_segment(text, node) or ""
                item = {"name": node.name, "line": node.lineno, "signature": self._signature(node), "doc": ast.get_docstring(node) or "", "source": source}
                functions.append(item)
                if relevance.search(node.name + " " + source):
                    relevant_chunks.append({"chunk_id": f"function:{node.name}:{node.lineno}", "heading": f"function {node.name}", "text": source[: self.chunk_chars]})
            elif isinstance(node, ast.ClassDef):
                source = ast.get_source_segment(text, node) or ""
                classes.append({"name": node.name, "line": node.lineno, "doc": ast.get_docstring(node) or "", "source": source})
                if relevance.search(node.name + " " + source):
                    relevant_chunks.append({"chunk_id": f"class:{node.name}:{node.lineno}", "heading": f"class {node.name}", "text": source[: self.chunk_chars]})
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        if not relevant_chunks:
            relevant_chunks = self._chunk_document(text)
        return {
            "source": str(path), "kind": "python", "module_docstring": ast.get_docstring(tree) or "",
            "functions": functions, "classes": classes, "imports": sorted(set(imports)),
            "has_algorithm_interface": all(name in {f["name"] for f in functions} for name in ("train", "predict", "evaluate")),
            "content_chunks": relevant_chunks, "character_count": len(text),
        }

    def extract_report(self, path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        validation = payload.get("validation", payload)
        return {
            "source": str(path), "kind": "validation_report", "run_id": payload.get("run_id"),
            "algorithm": validation.get("algorithm"), "status": validation.get("status"),
            "metrics": validation.get("metrics", {}), "runtime_seconds": validation.get("runtime_seconds"),
            "resource_usage": validation.get("resource_usage", {}), "errors": validation.get("errors", []),
            "repair_history": payload.get("repair_history", []),
            "content_chunks": self._chunk_document(json.dumps(payload, ensure_ascii=False, indent=2)),
        }

    def ingest(self, path: str | Path, store: KnowledgeStore) -> dict[str, Any]:
        path = Path(path)
        if path.suffix.lower() == ".py":
            deterministic = self.extract_python(path)
        elif path.suffix.lower() == ".json":
            deterministic = self.extract_report(path)
        else:
            deterministic = self.extract_markdown(path)
        result, extraction_trace = self.semantic_agent.run(path, deterministic)
        result["extraction_trace"] = extraction_trace
        result["deterministic_facts"] = {k: v for k, v in deterministic.items() if k != "content_chunks"}
        source_id = "source_" + self._slug(path.stem) + "_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
        result["id"] = source_id
        store.upsert_knowledge_item(source_id, "SourceDocument", result)
        self._materialize(result, source_id, store)
        if deterministic.get("has_algorithm_interface"):
            algorithm_id = "algorithm_extracted_" + self._slug(path.stem)
            store.upsert_algorithm({"id": algorithm_id, "name": path.stem, "task_types": ["binary_classification"], "source": str(path), "extracted_functions": deterministic["functions"], "imports": deterministic["imports"], "historical_metrics": {}, "provenance": {"source": str(path)}})
            store.add_edge(source_id, algorithm_id, "SUPPORTS", {"source": str(path), "kind": "ast"})
        store.export_graph()
        return result

    def _materialize(self, result: dict[str, Any], source_id: str, store: KnowledgeStore) -> None:
        id_map: dict[str, str] = {}
        for entity in result.get("entities", []):
            local_id = str(entity.get("id", entity.get("name", "entity")))
            node_type = entity.get("type", "SourceDocument")
            global_id = self._node_id(node_type, entity.get("name", local_id))
            id_map[local_id] = global_id
            payload = {"id": global_id, "name": entity.get("name", local_id), **entity.get("properties", {}), "confidence": entity.get("confidence"), "provenance": entity.get("provenance", {"source": result.get("provenance", {}).get("source"), "evidence_span": entity.get("evidence_span")})}
            if node_type == "Algorithm":
                store.upsert_algorithm({"task_types": payload.get("task_types", ["binary_classification"]), "historical_metrics": payload.get("historical_metrics", {}), **payload})
            elif node_type == "Capability":
                store.upsert_capability({"task_type": payload.get("task_type", "binary_classification"), **payload})
            elif node_type == "ValidationRun":
                store.add_validation_run({"run_id": global_id, "status": payload.get("status", "unknown"), "metrics": payload.get("metrics", {}), **payload})
            elif node_type == "FailureExperience":
                store.add_experience({"id": global_id, "kind": "extracted_failure", **payload})
            elif node_type == "RepairExperience":
                store.add_repair_experience(payload)
            elif node_type == "AlgorithmVersion":
                store.add_algorithm_version(payload)
            else:
                store.upsert_knowledge_item(global_id, node_type, payload)
            store.add_edge(source_id, global_id, "SUPPORTS", {"source": result.get("provenance", {}).get("source"), "evidence_span": entity.get("evidence_span"), "confidence": entity.get("confidence")})
        for relation in result.get("relations", []):
            source = id_map.get(str(relation.get("source")))
            target = id_map.get(str(relation.get("target")))
            if source and target:
                store.add_edge(source, target, str(relation.get("relation", "RELATED_TO")).upper(), {"source": result.get("provenance", {}).get("source"), "evidence_span": relation.get("evidence_span"), "confidence": relation.get("confidence")})

        # Normalize semantically distinct field roles if an extracted document identifies them.
        for entity in result.get("entities", []):
            name = str(entity.get("name", ""))
            properties = entity.get("properties", {})
            if properties.get("role") == "target" or name.lower() in {"churn", "target", "label"}:
                entity_id = self._node_id("Target", name)
                store.upsert_knowledge_item(entity_id, "Target", {"id": entity_id, "name": name, "source": result.get("provenance", {}).get("source")})
                store.add_source_support(source_id, entity_id)

    def ingest_path(self, path: str | Path, store: KnowledgeStore) -> list[dict[str, Any]]:
        path = Path(path)
        if path.is_file():
            return [self.ingest(path, store)]
        if not path.is_dir():
            raise FileNotFoundError(path)
        results = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".py", ".md", ".markdown", ".txt", ".json"} and ".git" not in child.parts and "__pycache__" not in child.parts:
                try:
                    results.append(self.ingest(child, store))
                except (SyntaxError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    results.append({"source": str(child), "kind": "error", "error": f"{type(exc).__name__}: {exc}"})
        return results

    def _chunk_document(self, text: str) -> list[dict[str, Any]]:
        lines = text.splitlines()
        chunks, current, hierarchy, index = [], [], [], 0
        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                level, title = len(match.group(1)), match.group(2)
                hierarchy = hierarchy[: level - 1] + [title]
            if current and sum(len(x) + 1 for x in current) + len(line) > self.chunk_chars:
                chunks.append({"chunk_id": f"chunk-{index}", "heading": " > ".join(hierarchy), "text": "\n".join(current)})
                index += 1
                current = current[-3:]
            current.append(line)
        if current:
            chunks.append({"chunk_id": f"chunk-{index}", "heading": " > ".join(hierarchy), "text": "\n".join(current)})
        return chunks or [{"chunk_id": "chunk-0", "heading": "", "text": ""}]

    @staticmethod
    def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args]]
        return f"{node.name}({', '.join(args)})"

    @staticmethod
    def _slug(value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", "_", str(value).strip()).strip("_").lower()
        return normalized or hashlib.sha256(str(value).encode()).hexdigest()[:10]

    def _node_id(self, node_type: str, name: str) -> str:
        return f"{node_type.lower()}_{self._slug(name)}"
