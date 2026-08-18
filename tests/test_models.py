"""Tests for point-prediction model builders."""

from __future__ import annotations

import pytest
from sklearn.linear_model import ElasticNet, Ridge

from house_price_uncertainty.models import (
    build_elasticnet_pipeline,
    build_ridge_pipeline,
)


def test_ridge_pipeline_contains_preprocessing_and_model() -> None:
    pipeline = build_ridge_pipeline(alpha=1.0)

    assert list(pipeline.named_steps) == [
        "preprocessing",
        "model",
    ]

    assert isinstance(
        pipeline.named_steps["model"],
        Ridge,
    )


def test_ridge_pipeline_uses_requested_alpha() -> None:
    pipeline = build_ridge_pipeline(alpha=2.5)

    model = pipeline.named_steps["model"]

    assert model.alpha == pytest.approx(2.5)


def test_ridge_pipeline_rejects_negative_alpha() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        build_ridge_pipeline(alpha=-1.0)


def test_elasticnet_pipeline_contains_preprocessing_and_model() -> None:
    pipeline = build_elasticnet_pipeline(
        alpha=1.0,
        l1_ratio=0.5,
    )

    assert list(pipeline.named_steps) == [
        "preprocessing",
        "model",
    ]

    assert isinstance(
        pipeline.named_steps["model"],
        ElasticNet,
    )


def test_elasticnet_pipeline_uses_requested_parameters() -> None:
    pipeline = build_elasticnet_pipeline(
        alpha=0.25,
        l1_ratio=0.75,
    )

    model = pipeline.named_steps["model"]

    assert model.alpha == pytest.approx(0.25)
    assert model.l1_ratio == pytest.approx(0.75)


def test_elasticnet_pipeline_rejects_nonpositive_alpha() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        build_elasticnet_pipeline(
            alpha=0.0,
            l1_ratio=0.5,
        )


def test_elasticnet_pipeline_rejects_invalid_l1_ratio() -> None:
    with pytest.raises(
        ValueError,
        match="l1_ratio",
    ):
        build_elasticnet_pipeline(
            alpha=1.0,
            l1_ratio=0.0,
        )