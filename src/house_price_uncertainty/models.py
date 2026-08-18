"""Point-prediction model builders."""

from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from house_price_uncertainty.preprocessing import build_primary_preprocessor


def build_ridge_pipeline(*, alpha: float = 1.0) -> Pipeline:
    """Build a leakage-safe Ridge regression pipeline."""
    if alpha < 0:
        raise ValueError("Ridge alpha must be non-negative.")

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_primary_preprocessor(
                    scale_numeric=True,
                ),
            ),
            (
                "model",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )