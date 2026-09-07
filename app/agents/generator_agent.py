from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib

from app.generation.templates import render_algorithm
from app.generation.task_contracts import build_codegen_prompt
from app.generation.code_ir import plan_to_code_ir, compile_code_ir
from app.llm.contracts import extract_python_code
from app.models import AlgorithmPlan, CapabilitySpec
from app.validation.checks import static_check
from app.validation.protocol import AlgorithmProtocolValidator


class GeneratorAgent:
    def __init__(self, llm=None, provider_name: str = "mock"):
        self.llm = llm
        self.provider_name = provider_name

    def run(self, run_dir: str | Path, spec: CapabilitySpec, plan: AlgorithmPlan, allow_llm: bool = True, mode: str = "free_form_llm") -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "algorithm.py"
        source = None
        code_ir = None
        generation_trace = {"provider": self.provider_name, "status": "template_fallback", "reason": "no usable LLM code"}
        if mode == "structured_synthesis":
            code_ir = plan_to_code_ir(spec, plan)
            source = compile_code_ir(spec, code_ir)
            generation_trace = {"provider": "deterministic_code_ir_compiler", "status": "structured_code_ir_compiled", "code_ir": code_ir.model_dump()}
        if mode == "free_form_llm" and allow_llm and self.llm is not None and self.provider_name != "mock":
            prompt = "只输出完整 Python 代码，不要思考过程、不要 Markdown。" + build_codegen_prompt(spec, plan)
            previous_output, gate_error, attempts = "", "", []
            for attempt in range(1, 3):
                try:
                    current_prompt = prompt
                    if attempt > 1:
                        current_prompt += f"\n上一次代码被严格门禁拒绝：{gate_error}。请返回修正后的完整代码。特别注意签名必须逐字是 def train(train_df, target_col, config=None)。\n上一次输出：\n{previous_output[:10000]}"
                    raw_output = self.llm.complete("你是安全的算法代码生成器。只输出完整代码。", current_prompt, purpose="code_generation")
                    previous_output = raw_output
                    proposed = extract_python_code(raw_output)
                    gate = static_check_text(proposed, spec.task_type, spec.target_column) if proposed else {"passed": False, "message": "no complete Python module extracted"}
                    if proposed and gate["passed"]:
                        source = proposed
                        attempts.append({"attempt": attempt, "status": "accepted"})
                        generation_trace = {"provider": getattr(self.llm, "last_provider", self.provider_name), "model": getattr(self.llm, "model", None), "status": "llm_code_accepted", "attempts": attempts, "token_usage": getattr(self.llm, "last_usage", {}), "generation": getattr(self.llm, "last_generation", {})}
                        break
                    gate_error = gate["message"]
                    attempts.append({"attempt": attempt, "status": "rejected", "reason": gate_error, "generation": getattr(self.llm, "last_generation", {})})
                except Exception as exc:
                    gate_error = f"{type(exc).__name__}: {exc}"
                    attempts.append({"attempt": attempt, "status": "provider_error", "reason": gate_error})
                    break
            if source is None:
                generation_trace.update({"reason": gate_error or "no usable LLM code", "attempts": attempts, "generation": getattr(self.llm, "last_generation", {}), "output_preview": previous_output[:1000]})
        if not allow_llm:
            generation_trace = {"provider": self.provider_name, "status": "template_by_budget", "reason": "LLM code budget reserved for top beam candidate"}
        path.write_text(source or render_algorithm(spec, plan), encoding="utf-8")
        metadata = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "spec": spec.to_dict(),
            "plan": plan.to_dict(),
            "code_ir": code_ir.model_dump() if code_ir else None,
            "generation": generation_trace,
        }
        (run_dir / "algorithm_meta.json").write_text(__import__("json").dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def static_check_text(source: str, task_type: str = "binary_classification", target_column: str = "") -> dict:
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"passed": False, "message": str(exc)}
    protocol = AlgorithmProtocolValidator().validate_source(source)
    if not protocol.passed:
        return {"passed": False, "message": protocol.to_dict()["message"]}
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
    from app.validation.semantic import GeneratedCodeSemanticValidator
    semantic = GeneratedCodeSemanticValidator().validate_source(source, task_type, target_column)
    if not semantic.passed:
        return semantic.to_dict()
    return {"passed": True, "message": "ok"}
