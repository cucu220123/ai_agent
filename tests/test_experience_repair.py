from app.agents.critic_agent import CriticAgent
from app.experience.retriever import ExperienceRetriever
from app.models import AlgorithmPlan, CapabilitySpec, ValidationResult


def test_similar_repair_experience_enters_critic():
    spec = CapabilitySpec(raw_description="churn", domain="customer_churn", task_type="binary_classification", target_column="churn")
    experiences = [{"id": "failure_interface_1", "task_type": "binary_classification", "domain": "customer_churn", "failure_type": "interface_failure", "triggering_condition": "missing predict", "repair_history": [{"changes": ["restore predict"]}], "repair_success": True}]
    retrieved = ExperienceRetriever().retrieve_failures(spec, experiences)
    assert retrieved[0]["id"] == "failure_interface_1"
    diagnosis = CriticAgent().run(spec, AlgorithmPlan("a", "A", "", []), ValidationResult(status="failed", algorithm="A", errors=["missing function: predict"]), "", retrieved)
    assert "failure_interface_1" in diagnosis["retrieved_experience_ids"]

