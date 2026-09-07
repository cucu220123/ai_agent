from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.generation.templates import render_algorithm
from app.models import AlgorithmPlan, CapabilitySpec


class FeatureHandlingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drop_target_before_feature_detection: bool = True
    numeric_imputer: str = "median"
    categorical_imputer: str = "most_frequent"
    encoder: str = "one_hot"
    scaler: str = "none"


class PredictionSpec(BaseModel):
    mode: Literal["probability", "label", "anomaly_score"] = "probability"
    threshold: float = Field(default=0.5, ge=0, le=1)


class AlgorithmImplementationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: str
    algorithm: str
    feature_handling: FeatureHandlingSpec
    estimator_params: dict[str, Any] = Field(default_factory=dict)
    prediction: PredictionSpec
    metrics: list[str]
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


def plan_to_code_ir(spec: CapabilitySpec, plan: AlgorithmPlan) -> AlgorithmImplementationSpec:
    params = dict(plan.hyperparameters)
    return AlgorithmImplementationSpec(
        task_type=spec.task_type,
        algorithm=plan.base_algorithm_id or plan.algorithm_id,
        feature_handling=FeatureHandlingSpec(
            drop_target_before_feature_detection=True,
            numeric_imputer=str(params.get("numeric_imputer", "median")),
            categorical_imputer="most_frequent",
            encoder="one_hot",
            scaler=str(params.get("scaler", "none")),
        ),
        estimator_params=params,
        prediction=PredictionSpec(mode="anomaly_score" if spec.task_type == "anomaly_detection" else "probability" if spec.probability_output_required else "label", threshold=float(params.get("threshold", 0.5))),
        metrics=spec.metrics,
        rationale=plan.rationale,
        evidence_ids=plan.evidence_ids,
    )


def compile_code_ir(spec: CapabilitySpec, implementation: AlgorithmImplementationSpec) -> str:
    plan = AlgorithmPlan(
        algorithm_id=f"algorithm_{implementation.algorithm}", algorithm_name=implementation.algorithm,
        rationale=implementation.rationale,
        preprocessing=[implementation.feature_handling.numeric_imputer, implementation.feature_handling.categorical_imputer, implementation.feature_handling.encoder, implementation.feature_handling.scaler],
        hyperparameters=implementation.estimator_params,
        base_algorithm_id=f"algorithm_{implementation.algorithm}",
        preprocessing_variant=f"imputer={implementation.feature_handling.numeric_imputer};scaler={implementation.feature_handling.scaler}",
        config_variant="code_ir",
    )
    return render_algorithm(spec, plan)

