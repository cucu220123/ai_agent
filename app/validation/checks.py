from __future__ import annotations

import ast
import importlib.util
import time
from pathlib import Path


ALLOWED_IMPORT_ROOTS = {"__future__", "typing", "numpy", "pandas", "sklearn"}
BLOCKED_NAMES = {"eval", "exec", "compile", "__import__", "open", "input"}
BLOCKED_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "ctypes"}


def static_check(path: str | Path) -> dict:
    path = Path(path)
    code = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"passed": False, "message": f"syntax error: {exc}"}
    violations: list[str] = []
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    violations.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                violations.append(f"disallowed import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
            violations.append(f"blocked call: {node.func.id}")
        elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_NAMES:
            violations.append(f"blocked attribute: {node.attr}")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_MODULES:
            violations.append(f"blocked name: {node.id}")
    missing = {"train", "predict", "evaluate"} - functions
    violations.extend(f"missing function: {name}" for name in sorted(missing))
    return {"passed": not violations, "message": "; ".join(violations) if violations else "static checks passed"}


def import_check(path: str | Path) -> dict:
    path = Path(path)
    started = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("generated_algorithm", path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot create import spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = [name for name in ("train", "predict", "evaluate") if not hasattr(module, name)]
        return {"passed": not missing, "message": "import and interface checks passed" if not missing else f"missing interface: {missing}", "module": module, "duration": time.perf_counter() - started}
    except Exception as exc:
        return {"passed": False, "message": f"import error: {type(exc).__name__}: {exc}", "duration": time.perf_counter() - started}
