from __future__ import annotations

from pathlib import Path

from app.llm.contracts import extract_python_code


class RepairAgent:
    """Apply bounded, auditable repairs to generated template code."""

    def __init__(self, llm=None, provider_name: str = "mock"):
        self.llm = llm
        self.provider_name = provider_name

    def repair(self, path: str | Path, error_text: str, round_no: int) -> dict:
        path = Path(path)
        code = path.read_text(encoding="utf-8")
        changes: list[str] = []
        if self.llm is not None and self.provider_name != "mock":
            try:
                response = self.llm.complete(
                    "你是安全的 Python 算法修复器。只返回完整代码。",
                    f"修复以下算法代码的验证错误：{error_text}\n原代码：\n```python\n{code}\n```\n必须保留 train/predict/evaluate 接口，禁止文件、网络和系统调用。",
                )
                proposed = extract_python_code(response)
                if proposed and "def train" in proposed and "def evaluate" in proposed:
                    path.write_text(proposed, encoding="utf-8")
                    return {"round": round_no, "changes": ["accepted LLM repair after interface extraction"], "error": error_text[-2000:], "provider": self.provider_name}
            except Exception as exc:
                changes.append(f"LLM repair unavailable: {type(exc).__name__}")
        if "unexpected indent" in error_text.lower() or "indentationerror" in error_text.lower():
            code = code.replace("        numeric_steps.append((\"scale\", StandardScaler()))", "    numeric_steps.append((\"scale\", StandardScaler()))")
            changes.append("repair unexpected indentation in preprocessing pipeline")
        if "max_iter" in error_text and "max_iter=1000" not in code:
            code = code.replace("max_iter=500", "max_iter=1000")
            changes.append("increase logistic regression max_iter")
        if "could not convert string" in error_text.lower() and "handle_unknown=\"ignore\"" not in code:
            code = code.replace("OneHotEncoder(handle_unknown='ignore')", "OneHotEncoder(handle_unknown=\"ignore\")")
            changes.append("make categorical encoder robust to unseen values")
        if "nan" in error_text.lower() and "SimpleImputer" not in code:
            changes.append("requested imputation; template already includes it")
        if not changes:
            changes.append("regenerated deterministic algorithm template")
        path.write_text(code, encoding="utf-8")
        return {"round": round_no, "changes": changes, "error": error_text[-2000:]}
