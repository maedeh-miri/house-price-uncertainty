"""Point-prediction model builders."""

from __future__ import annotations

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline

from house_price_uncertainty.preprocessing import build_primary_preprocessor

ELASTIC_NET_MAX_ITER = 50_000


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


def build_elasticnet_pipeline(
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> Pipeline:
    """Build a leakage-safe ElasticNet regression pipeline."""
    if alpha <= 0:
        raise ValueError("ElasticNet alpha must be positive.")

    if not 0.0 < l1_ratio <= 1.0:
        raise ValueError(
            "ElasticNet l1_ratio must be greater than 0 and at most 1."
        )

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
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=ELASTIC_NET_MAX_ITER,
                ),
            ),
        ]
    )