"""Shared source safety and interface gate for generation, repair and execution."""
from __future__ import annotations
import ast
from pathlib import Path
from app.validation.protocol import AlgorithmProtocolValidator

ALLOWED_IMPORT_ROOTS = {"__future__", "typing", "numpy", "pandas", "sklearn"}
BLOCKED_NAMES = {"eval", "exec", "compile", "__import__", "open", "input", "globals", "locals", "vars", "breakpoint"}
BLOCKED_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "ctypes"}
BLOCKED_ATTRIBUTES = {"load", "loads", "save", "savez", "dump", "dumps", "to_csv", "to_pickle", "to_json", "to_sql", "to_excel", "fromfile", "tofile", "memmap", "ctypes", "ctypeslib", "system", "popen", "urlopen"}


def static_check_source(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"passed": False, "category": "syntax", "message": f"syntax error: {exc}"}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                    violations.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                violations.append(f"disallowed import: {node.module}")
            if any(alias.name == "*" for alias in node.names):
                violations.append("wildcard imports are disallowed")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES | BLOCKED_MODULES:
            violations.append(f"blocked name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith(("__", "read_", "fetch_")) or node.attr in BLOCKED_ATTRIBUTES | BLOCKED_NAMES:
                violations.append(f"blocked attribute: {node.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"getattr", "setattr", "delattr"}:
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str) or node.args[1].value.startswith("__"):
                violations.append("dynamic/dunder reflection is disallowed")
    if violations:
        return {"passed": False, "category": "security", "message": "; ".join(dict.fromkeys(violations))}
    protocol = AlgorithmProtocolValidator().validate_source(code)
    return {"passed": protocol.passed, "category": "interface" if not protocol.passed else "ok", "message": protocol.to_dict()["message"]}


def static_check(path: str | Path) -> dict:
    return static_check_source(Path(path).read_text(encoding="utf-8"))


def import_check(path: str | Path) -> dict:
    return AlgorithmProtocolValidator().validate_path(path).to_dict()

