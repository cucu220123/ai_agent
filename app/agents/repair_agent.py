from __future__ import annotations

from pathlib import Path


class RepairAgent:
    """Apply bounded, auditable repairs to generated template code."""

    def repair(self, path: str | Path, error_text: str, round_no: int) -> dict:
        path = Path(path)
        code = path.read_text(encoding="utf-8")
        changes: list[str] = []
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
