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


def test_planner_graph_ids_resolve_to_registered_algorithm_identity():
    value = PlannerAdviceContract.normalize_candidate({"candidate_algorithms": ["algorithm_random_forest"], "algorithm_reasons": {"algorithm_random_forest": "Evidence supports this model"}, "preprocessing_recommendations": {"algorithm_random_forest": ["median imputation"]}, "hyperparameter_recommendations": {"algorithm_random_forest": {"max_depth": 5}}})
    assert value["candidate_algorithms"] == ["random_forest"]
    assert value["hyperparameter_recommendations"]["random_forest"]["max_depth"] == 5
    assert "random_forest" in value["algorithm_reasons"]


def test_business_labels_become_executable_contract_without_target_leakage(tmp_path):
    data = tmp_path / "input.csv"
    data.write_text("age,churn\n20,0\n30,1\n40,0\n50,1\n")
    class BusinessLLM:
        def complete(self, *args, **kwargs):
            return json.dumps({"domain": "customer_churn", "capability_name": "Customer churn", "task_type": "binary_classification", "data_type": "tabular", "target": "churn", "input_schema": {"age": "int", "churn": "int"}, "output_schema": {"predicted_churn": "int", "positive_class_probability": "float"}, "metrics": ["roc_auc"], "probability_output_required": True, "confidence": 0.95})
    spec, trace = RequirementUnderstandingAgent(BusinessLLM(), "openai").run("Customer churn ROC-AUC", data)
    assert trace["status"] == "ok"
    assert set(spec.output_schema) == {"prediction", "probability"}
    assert "churn" not in spec.input_schema
    assert trace["business_output_schema"]["predicted_churn"] == "int"
