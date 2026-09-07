from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from app.knowledge.store import KnowledgeStore
from app.agents.knowledge_extraction_agent import KnowledgeExtractionAgent


class CapabilityExtractor:
    """Extract lightweight capability/algorithm facts from Markdown and Python files."""

    def __init__(self, llm=None, provider_name: str = "mock"):
        self.semantic_agent = KnowledgeExtractionAgent(llm, provider_name)

    def extract_markdown(self, path: str | Path) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        headings = re.findall(r"^#+\s+(.+)$", text, flags=re.MULTILINE)
        metrics = sorted(set(re.findall(r"\b(?:ROC-AUC|PR-AUC|F1|Precision|Recall|RMSE|MAE|R2)\b", text, flags=re.I)))
        fields = re.findall(r"^\|\s*([A-Za-z_][\w]*)\s*\|", text, flags=re.MULTILINE)
        return {"source": str(path), "kind": "markdown", "headings": headings, "metrics": metrics, "fields": fields, "summary": text[:1000]}

    def extract_python(self, path: str | Path) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(text)
        functions = []
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append({"name": node.name, "line": node.lineno, "doc": ast.get_docstring(node) or ""})
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        return {"source": str(path), "kind": "python", "functions": functions, "imports": sorted(set(imports)), "has_algorithm_interface": all(name in {f["name"] for f in functions} for name in ("train", "predict", "evaluate"))}

    def ingest(self, path: str | Path, store: KnowledgeStore) -> dict[str, Any]:
        path = Path(path)
        deterministic = self.extract_python(path) if path.suffix == ".py" else self.extract_markdown(path)
        result, extraction_trace = self.semantic_agent.run(path, deterministic)
        result["extraction_trace"] = extraction_trace
        source_id = "source_" + re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_").lower()
        result["id"] = source_id
        store.upsert_knowledge_item(source_id, "SourceDocument", result)
        self._materialize_structured(result, source_id, store)
        for capability in result.get("capabilities", []):
            if isinstance(capability, dict) and capability.get("id"):
                store.add_source_support(source_id, capability["id"])
        for algorithm in result.get("algorithms", []):
            if isinstance(algorithm, dict) and algorithm.get("id"):
                store.add_source_support(source_id, algorithm["id"])
        if result.get("has_algorithm_interface"):
            algorithm_id = "algorithm_extracted_" + source_id.removeprefix("source_")
            store.upsert_algorithm({"id": algorithm_id, "name": path.stem, "task_types": ["binary_classification"], "source": str(path), "extracted_functions": result["functions"], "imports": result["imports"], "historical_metrics": {}})
            store.add_edge(source_id, algorithm_id, "EXTRACTED_ALGORITHM")
        store.export_graph()
        return result

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip()).strip("_").lower() or "item"

    def _materialize_structured(self, result: dict[str, Any], source_id: str, store: KnowledgeStore) -> None:
        """Turn LLM extraction arrays into typed nodes and provenance edges."""
        plural_types = {
            "tasks": "Task", "datasets": "Dataset", "features": "Feature", "preprocessing": "PreprocessingStrategy",
            "metrics": "Metric", "constraints": "Constraint", "dependencies": "Dependency", "hyperparameters": "HyperparameterConfig",
        }
        for key, node_type in plural_types.items():
            values = result.get(key, [])
            if isinstance(values, (str, dict)):
                values = [values]
            for index, value in enumerate(values or []):
                payload = value if isinstance(value, dict) else {"name": value}
                name = payload.get("id") or payload.get("name") or payload.get("metric") or f"{key}_{index}"
                node_id = f"{node_type.lower()}_{self._slug(name)}"
                payload = {"id": node_id, **payload, "provenance": payload.get("provenance", {"source": result.get("source")})}
                store.upsert_knowledge_item(node_id, node_type, payload)
                store.add_source_support(source_id, node_id)
        for index, value in enumerate(result.get("algorithms", []) or []):
            payload = value if isinstance(value, dict) else {"name": value}
            name = payload.get("id") or payload.get("name") or f"algorithm_{index}"
            algorithm_id = f"algorithm_extracted_{self._slug(name)}"
            payload = {"id": algorithm_id, "name": name, "task_types": payload.get("task_types", ["binary_classification"]), "historical_metrics": payload.get("historical_metrics", {}), **payload}
            store.upsert_algorithm(payload)
            store.add_source_support(source_id, algorithm_id)
        for index, value in enumerate(result.get("capabilities", []) or []):
            payload = value if isinstance(value, dict) else {"name": value}
            name = payload.get("id") or payload.get("name") or f"capability_{index}"
            capability_id = f"cap_extracted_{self._slug(name)}"
            payload = {"id": capability_id, "name": name, "task_type": payload.get("task_type", "binary_classification"), **payload}
            store.upsert_capability(payload)
            store.add_source_support(source_id, capability_id)
        for index, value in enumerate(result.get("validation_results", []) or []):
            payload = value if isinstance(value, dict) else {"summary": value}
            run_id = payload.get("run_id") or f"extracted_run_{self._slug(Path(result.get('source', 'source')).stem)}_{index}"
            store.add_validation_run({"run_id": run_id, "status": payload.get("status", "unknown"), "algorithm_id": payload.get("algorithm_id"), "metrics": payload.get("metrics", payload.get("scores", {})), "timestamp": payload.get("timestamp"), "source_id": source_id, **payload})
        for index, value in enumerate(result.get("failure_cases", []) or []):
            payload = value if isinstance(value, dict) else {"summary": value}
            failure_id = payload.get("id") or f"extracted_failure_{self._slug(Path(result.get('source', 'source')).stem)}_{index}"
            store.add_experience({"id": failure_id, "kind": "extracted_failure", "source_id": source_id, **payload})

    def ingest_path(self, path: str | Path, store: KnowledgeStore) -> list[dict[str, Any]]:
        path = Path(path)
        if path.is_file():
            return [self.ingest(path, store)]
        if not path.is_dir():
            raise FileNotFoundError(path)
        results = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".py", ".md", ".markdown"} and ".git" not in child.parts and "__pycache__" not in child.parts:
                try:
                    results.append(self.ingest(child, store))
                except (SyntaxError, UnicodeDecodeError) as exc:
                    results.append({"source": str(child), "kind": "error", "error": f"{type(exc).__name__}: {exc}"})
        return results
