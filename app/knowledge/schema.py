"""Canonical schema for the algorithm capability graph."""

from enum import Enum


class RelationType(str, Enum):
    SOLVES = "SOLVES"
    HAS_INPUT = "HAS_INPUT"
    PREDICTS = "PREDICTS"
    ACCEPTS = "ACCEPTS"
    OUTPUTS = "OUTPUTS"
    USES_ALGORITHM = "USES_ALGORITHM"
    SUITABLE_FOR = "SUITABLE_FOR"
    USES_PREPROCESSING = "USES_PREPROCESSING"
    EVALUATED_BY = "EVALUATED_BY"
    VALIDATED_ON = "VALIDATED_ON"
    ON_DATASET = "ON_DATASET"
    HAS_CONFIG = "HAS_CONFIG"
    REQUIRES = "REQUIRES"
    OCCURRED_IN = "OCCURRED_IN"
    REPAIRS = "REPAIRS"
    SUPPORTS = "SUPPORTS"
    VERSION_OF = "VERSION_OF"
    VALIDATES = "VALIDATES"
    REQUIRES_FEATURE = "REQUIRES_FEATURE"
    RELATED_TO = "RELATED_TO"

NODE_TYPES = [
    "Capability",
    "Task",
    "Algorithm",
    "AlgorithmVersion",
    "Dataset",
    "FeatureStrategy",
    "Feature",
    "Target",
    "InputSchema",
    "OutputSchema",
    "Metric",
    "Constraint",
    "Dependency",
    "Environment",
    "HyperparameterConfig",
    "ValidationRun",
    "FailureExperience",
    "RepairExperience",
    "SourceDocument",
]

EDGE_TYPES = [
    "USES_ALGORITHM",
    "SOLVES",
    "SUITABLE_FOR",
    "VALIDATED_ON",
    "ON_DATASET",
    "REQUIRES_FEATURE",
    "USES_PREPROCESSING",
    "EVALUATED_BY",
    "REQUIRES",
    "HAS_CONFIG",
    "VALIDATES",
    "OCCURRED_IN",
    "REPAIRS",
    "SUPPORTS",
    "VERSION_OF",
    "RELATED_TO",
    "FIXED_BY",
]

EDGE_TYPES = [relation.value for relation in RelationType]
