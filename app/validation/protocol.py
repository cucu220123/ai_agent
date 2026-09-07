from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProtocolValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "errors": self.errors, "message": "; ".join(self.errors) if self.errors else "algorithm protocol passed"}


class AlgorithmProtocolValidator:
    """Validate the exact versioned callable contract before importing generated code."""

    signatures = {
        "train": (["train_df", "target_col", "config"], 1),
        "predict": (["model", "test_df"], 0),
        "evaluate": (["model", "test_df", "target_col"], 0),
    }

    def validate_path(self, path: str | Path) -> ProtocolValidation:
        try:
            return self.validate_source(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            return ProtocolValidation(False, [f"source error: {type(exc).__name__}: {exc}"])

    def validate_source(self, source: str) -> ProtocolValidation:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return ProtocolValidation(False, [f"syntax error: {exc}"])
        errors: list[str] = []
        async_names = {node.name for node in tree.body if isinstance(node, ast.AsyncFunctionDef)}
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in self.signatures:
            if name in async_names:
                errors.append(f"{name} must not be async")
            if name not in functions:
                errors.append(f"missing function: {name}")
                continue
            node = functions[name]
            expected_names, expected_defaults = self.signatures[name]
            actual_names = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args]]
            if actual_names != expected_names:
                errors.append(f"{name} signature must be ({', '.join(expected_names)}), got ({', '.join(actual_names)})")
            if len(node.args.defaults) != expected_defaults:
                errors.append(f"{name} must have {expected_defaults} optional positional argument(s)")
            if name == "train" and node.args.defaults:
                default = node.args.defaults[-1]
                if not isinstance(default, ast.Constant) or default.value is not None:
                    errors.append("train config default must be None")
            if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                errors.append(f"{name} must not use *args, **kwargs, or keyword-only arguments")
        return ProtocolValidation(not errors, errors)

