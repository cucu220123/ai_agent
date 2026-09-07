import json

import pytest
from pydantic import ValidationError

from app.agents.advisor_agent import AdvisorAgent, PlannerAdviceContract
from app.agents.requirement_agent import RequirementContract, RequirementUnderstandingAgent
from app.models import CapabilitySpec, KnowledgeContext
from app.validation.protocol import AlgorithmProtocolValidator


class IrrelevantLLM:
    model = "irrelevant"
    last_usage = {}
    last_generation = {"finish_reason": "stop", "truncated": False}

    def complete(self, system, user, purpose="general", generation_config=None):
        return json.dumps({"prediction": 0, "probability": 0.95})


def test_wrong_requirement_json_is_rejected():
    with pytest.raises(ValidationError):
        RequirementContract.model_validate({"prediction": 0, "probability": 0.95})


def test_irrelevant_requirement_response_falls_back(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("age,churn\n20,0\n30,1\n", encoding="utf-8")
    spec, trace = RequirementUnderstandingAgent(IrrelevantLLM(), "local").run("预测客户流失，要求 ROC-AUC 0.75", data)
    assert trace["status"] == "fallback"
    assert not trace["schema_valid"]
    assert len(trace["attempts"]) == 2
    assert spec.target_column == "churn"


def test_wrong_planner_json_is_rejected():
    with pytest.raises(ValidationError):
        PlannerAdviceContract.model_validate({"prediction": 0, "probability": 0.95})
    advice, trace = AdvisorAgent(IrrelevantLLM()).run(CapabilitySpec(raw_description="x"), KnowledgeContext(algorithms=[{"id": "algorithm_logistic_regression"}]))
    assert trace["status"] == "fallback"
    assert not trace["schema_valid"]
    assert advice["risks"]


def test_exact_algorithm_protocol_rejects_extra_train_args():
    source = """
def train(train_df, target_col, C, max_iter): pass
def predict(model, test_df): pass
def evaluate(model, test_df, target_col): pass
"""
    result = AlgorithmProtocolValidator().validate_source(source)
    assert not result.passed
    assert "train signature must be" in result.to_dict()["message"]

