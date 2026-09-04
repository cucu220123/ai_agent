from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from app.knowledge.store import KnowledgeStore


class CapabilityExtractor:
    """Extract lightweight capability/algorithm facts from Markdown and Python files."""

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
        result = self.extract_python(path) if path.suffix == ".py" else self.extract_markdown(path)
        source_id = "source_" + re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_").lower()
        result["id"] = source_id
        store.upsert_knowledge_item(source_id, "SourceDocument", result)
        if result.get("has_algorithm_interface"):
            algorithm_id = "algorithm_extracted_" + source_id.removeprefix("source_")
            store.upsert_algorithm({"id": algorithm_id, "name": path.stem, "task_types": ["binary_classification"], "source": str(path), "extracted_functions": result["functions"], "imports": result["imports"], "historical_metrics": {}})
            store.add_edge(source_id, algorithm_id, "EXTRACTED_ALGORITHM")
        store.export_graph()
        return result

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
