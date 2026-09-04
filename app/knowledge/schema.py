"""Canonical schema for the algorithm capability graph."""

NODE_TYPES = [
    "Capability",
    "Algorithm",
    "Dataset",
    "FeatureStrategy",
    "Metric",
    "Environment",
    "ValidationRun",
    "FailureExperience",
    "Constraint",
]

EDGE_TYPES = [
    "USES_ALGORITHM",
    "VALIDATED_ON",
    "REQUIRES_FEATURE",
    "EVALUATED_BY",
    "REQUIRES",
    "VALIDATES",
    "RELATED_TO",
    "FIXED_BY",
]

