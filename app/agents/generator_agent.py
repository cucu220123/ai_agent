from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib

from app.generation.templates import render_algorithm
from app.llm.contracts import extract_python_code
from app.models import AlgorithmPlan, CapabilitySpec
from app.validation.checks import static_check


class GeneratorAgent:
    def __init__(self, llm=None, provider_name: str = "mock"):
        self.llm = llm
        self.provider_name = provider_name

    def run(self, run_dir: str | Path, spec: CapabilitySpec, plan: AlgorithmPlan) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "algorithm.py"
        source = None
        generation_trace = {"provider": self.provider_name, "status": "template_fallback", "reason": "no usable LLM code"}
        if self.llm is not None and self.provider_name != "mock":
            prompt = (
                "生成一个仅依赖 pandas、numpy、scikit-learn 的客户流失二分类 Python 模块。"
                "必须只返回代码，且提供 train(train_df,target_col,config)、predict(model,test_df)、evaluate(model,test_df,target_col)。"
                f"算法方案：{plan.to_dict()}；任务规范：{spec.to_dict()}。禁止文件、网络、系统调用。"
            )
            try:
                source = extract_python_code(self.llm.complete("你是安全的算法代码生成器。", prompt))
                if source and static_check_text(source)["passed"]:
                    generation_trace = {"provider": self.provider_name, "status": "llm_code_accepted"}
                else:
                    source = None
                    generation_trace["reason"] = "LLM code failed extraction or static checks"
            except Exception as exc:
                generation_trace["reason"] = f"{type(exc).__name__}: {exc}"
        path.write_text(source or render_algorithm(spec, plan), encoding="utf-8")
        metadata = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "spec": spec.to_dict(),
            "plan": plan.to_dict(),
            "generation": generation_trace,
        }
        (run_dir / "algorithm_meta.json").write_text(__import__("json").dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def static_check_text(source: str) -> dict:
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"passed": False, "message": str(exc)}
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if not {"train", "predict", "evaluate"}.issubset(functions):
        return {"passed": False, "message": "missing required interfaces"}
    allowed = {"__future__", "typing", "numpy", "pandas", "sklearn"}
    blocked = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "ctypes"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    return {"passed": False, "message": f"disallowed import: {alias.name}"}
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in allowed:
            return {"passed": False, "message": f"disallowed import: {node.module}"}
        if isinstance(node, ast.Name) and node.id in blocked:
            return {"passed": False, "message": f"blocked name: {node.id}"}
    return {"passed": True, "message": "ok"}
