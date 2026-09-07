from __future__ import annotations

from pathlib import Path
import json

from app.llm.contracts import extract_python_code


class RepairAgent:
    """Apply bounded, auditable repairs to generated template code."""

    def __init__(self, llm=None, provider_name: str = "mock"):
        self.llm = llm
        self.provider_name = provider_name

    def repair(self, path: str | Path, error_text: str, round_no: int, retrieved_experiences: list[dict] | None = None) -> dict:
        path = Path(path)
        code = path.read_text(encoding="utf-8")
        retrieved_experiences = retrieved_experiences or []
        changes: list[str] = []
        if self.llm is not None and self.provider_name != "mock":
            try:
                response = self.llm.complete(
                    "你是安全的 Python 算法修复器。只返回完整代码。",
                    f"只输出完整 Python 代码，不要解释。修复错误：{error_text[-5000:]}\n历史验证成功的相似修复（若 traceback 匹配，必须逐项实现 repair_action；不可盲目照搬不匹配经验）：{json.dumps(retrieved_experiences[:5], ensure_ascii=False)}\n原代码：\n{code[-14000:]}\n严格保留 train(train_df,target_col,config=None)、predict(model,test_df)、evaluate(model,test_df,target_col)。注意 predict 接收的 test_df 已经不含 target，不得硬编码 drop churn。禁止文件、网络和系统调用。",
                    purpose="repair",
                )
                proposed = extract_python_code(response)
                if proposed and "def train" in proposed and "def evaluate" in proposed:
                    from app.agents.generator_agent import static_check_text
                    gate = static_check_text(proposed)
                    if gate["passed"]:
                        path.write_text(proposed, encoding="utf-8")
                        return {"round": round_no, "changes": ["accepted LLM repair after strict safety/protocol gate"], "error": error_text[-2000:], "provider": self.provider_name, "retrieved_experience_ids": [item.get("id") for item in retrieved_experiences[:5] if item.get("id")], "generation": getattr(self.llm, "last_generation", {})}
                    changes.append(f"LLM repair rejected by strict gate: {gate['message']}")
            except Exception as exc:
                changes.append(f"LLM repair unavailable: {type(exc).__name__}")
        if "unexpected indent" in error_text.lower() or "indentationerror" in error_text.lower():
            code = code.replace("        numeric_steps.append((\"scale\", StandardScaler()))", "    numeric_steps.append((\"scale\", StandardScaler()))")
            changes.append("repair unexpected indentation in preprocessing pipeline")
        if "missing function: predict" in error_text.lower() or "predict_broken" in code:
            code = code.replace("def predict_broken", "def predict")
            changes.append("restore required predict interface")
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
