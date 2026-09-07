"""Structured diagnosis/strategy/code repair with the same gate as initial generation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from app.llm.contracts import extract_json_object
from app.llm.security import sanitize


class RepairContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    failure_diagnosis: str = Field(min_length=5)
    repair_strategy: str = Field(min_length=5)
    revised_code: str = Field(min_length=80)


class RepairAgent:
    def __init__(self, llm=None, provider_name: str = "mock"):
        self.llm, self.provider_name = llm, provider_name

    def repair(self, path: str | Path, error_text: str, round_no: int, retrieved_experiences: list[dict] | None = None, task_type: str = "binary_classification", target_column: str = "") -> dict:
        from app.agents.generator_agent import static_check_text
        path = Path(path)
        code = path.read_text(encoding="utf-8")
        before_hash = hashlib.sha256(code.encode()).hexdigest()
        experiences = retrieved_experiences or []
        event: dict[str, Any] = {"round": round_no, "before_sha256": before_hash, "changes": [], "retrieved_experience_ids": [e.get("id") for e in experiences if e.get("id")], "error": error_text[-4000:], "attempts": []}
        if self.llm is not None and self.provider_name != "mock":
            payload = {"original_code": code, "validation_and_critic": error_text[-16000:], "retrieved_failure_and_repair_experience": experiences[:5], "task_type": task_type, "target": target_column, "json_schema": RepairContract.model_json_schema(), "rules": ["Return diagnosis, strategy and the complete revised_code in JSON.", "Preserve metadata() and exact train(train_df,target_col,config=None), predict(model,test_df), evaluate(model,test_df,target_col), predict_proba(model,test_df) if probability is required.", "predict receives features only. Exclude target before discovering training features.", "Never fake metrics: the trusted parent recomputes them.", "No I/O, network, processes or unsafe imports. Honor random_state from config."]}
            for attempt in range(2):
                try:
                    raw = self.llm.complete("You are RepairAgent. Diagnose the observed failure and return a complete minimal corrected implementation as strict JSON.", json.dumps(payload, ensure_ascii=False), purpose="repair", generation_config={"json_schema": RepairContract.model_json_schema()})
                    path.with_name(f"repair_round_{round_no}_attempt_{attempt + 1}.json").write_text(json.dumps({"raw_response": sanitize(raw)}, ensure_ascii=False, indent=2), encoding="utf-8")
                    repair = RepairContract.model_validate(extract_json_object(raw))
                    gate = static_check_text(repair.revised_code, task_type, target_column)
                    if not gate["passed"]:
                        raise ValueError(gate["message"])
                    if repair.revised_code == code:
                        raise ValueError("repair made no source change")
                    path.write_text(repair.revised_code, encoding="utf-8")
                    event.update(provider=getattr(self.llm, "last_provider", self.provider_name), status="llm_repair_accepted", diagnosis=repair.failure_diagnosis, strategy=repair.repair_strategy, changes=["accepted LLM repair after strict safety/protocol gate"], after_sha256=hashlib.sha256(repair.revised_code.encode()).hexdigest())
                    return sanitize(event)
                except Exception as exc:
                    message = sanitize(f"{type(exc).__name__}: {exc}")
                    event["attempts"].append({"attempt": attempt + 1, "status": "rejected", "error": message})
                    if os.getenv("AI_FACTORY_TRACE", "0") == "1":
                        print("[RepairAgent] gate rejected: " + message, flush=True)
                    payload["previous_error"] = message
            # Real mode leaves the failure visible. It never pretends a regex repair is LLM repair.
            event.update(provider=self.provider_name, status="repair_rejected", after_sha256=before_hash, changes=["no accepted LLM revision; source unchanged"])
            return sanitize(event)
        revised = code.replace("def predict_broken", "def predict")
        if revised != code:
            path.write_text(revised, encoding="utf-8")
            event["changes"] = ["offline fixture: restore predict interface"]
        else:
            event["changes"] = ["no applicable deterministic repair; source unchanged"]
        event.update(provider="mock", status="deterministic_repair", after_sha256=hashlib.sha256(revised.encode()).hexdigest())
        return sanitize(event)

