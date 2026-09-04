from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(path: str | Path, n_rows: int = 1200, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 76, n_rows)
    region = rng.choice(["north", "east", "south", "west"], n_rows, p=[0.25, 0.30, 0.25, 0.20])
    login_count = rng.poisson(12, n_rows).clip(0, 45)
    total_spend = np.maximum(rng.gamma(4.0, 180.0, n_rows), 5).round(2)
    complaint_count = rng.poisson(0.7, n_rows).clip(0, 8)
    membership = rng.choice(["basic", "silver", "gold", "platinum"], n_rows, p=[0.35, 0.30, 0.23, 0.12])
    tenure = rng.integers(1, 73, n_rows)
    logit = (
        0.9
        - 0.24 * login_count
        - 0.0035 * total_spend
        + 0.82 * complaint_count
        - 0.038 * tenure
        + 0.55 * (membership == "basic")
        + 0.38 * (region == "west")
        + 0.018 * np.maximum(age - 45, 0)
        + rng.normal(0, 0.28, n_rows)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    df = pd.DataFrame({
        "age": age, "region": region, "login_count_30d": login_count,
        "total_spend": total_spend, "complaint_count": complaint_count,
        "membership_level": membership, "tenure_months": tenure, "churn": churn,
    })
    # Deliberate missing values exercise the generated imputation pipeline.
    for column, fraction in (("total_spend", 0.025), ("region", 0.015), ("login_count_30d", 0.01)):
        indices = rng.choice(n_rows, size=max(1, int(n_rows * fraction)), replace=False)
        df.loc[indices, column] = np.nan
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/churn_demo.csv")
    parser.add_argument("--rows", type=int, default=1200)
    args = parser.parse_args()
    print(generate(args.output, args.rows))
