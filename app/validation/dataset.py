"""Measured dataset facts; language models never supply authoritative statistics."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def profile_dataset(path: str | Path, target: str = "") -> dict[str, Any]:
    path = Path(path)
    frame = pd.read_csv(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    features = [c for c in frame.columns if c != target]
    profile: dict[str, Any] = {
        "dataset_id": "dataset_" + digest[:16], "sha256": digest,
        "path": str(path), "rows": len(frame), "row_count": len(frame),
        "column_count": len(frame.columns), "columns": list(frame.columns),
        "feature_columns": features, "feature_count": len(features),
        "dtypes": {c: str(frame[c].dtype) for c in frame.columns},
        "missing_rates": {c: float(frame[c].isna().mean()) for c in frame.columns},
        "numeric_fraction": sum(pd.api.types.is_numeric_dtype(frame[c]) for c in features) / max(1, len(features)),
    }
    if target and target in frame:
        counts = frame[target].value_counts(dropna=False)
        profile["class_count"] = len(counts)
        if len(counts) <= 20:
            profile["class_balance"] = {str(k): float(v / len(frame)) for k, v in counts.items()}
            profile["minority_rate"] = float(counts.min() / len(frame))
        if len(counts) == 2:
            profile["positive_rate"] = float((frame[target] == sorted(counts.index, key=str)[-1]).mean())
    return profile
