"""Point-prediction model builders."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
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


def build_random_forest_pipeline(
    *,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: float | str | None = 1.0,
    random_state: int = 2026,
) -> Pipeline:
    """Build a leakage-safe Random Forest regression pipeline."""
    if n_estimators <= 0:
        raise ValueError("Random Forest n_estimators must be positive.")

    if max_depth is not None and max_depth <= 0:
        raise ValueError("Random Forest max_depth must be positive or None.")

    if min_samples_leaf <= 0:
        raise ValueError("Random Forest min_samples_leaf must be positive.")

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_primary_preprocessor(
                    scale_numeric=False,
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )