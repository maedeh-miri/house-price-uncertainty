"""Tests for point-prediction model builders."""

from __future__ import annotations

import pytest
from sklearn.linear_model import Ridge

from house_price_uncertainty.models import build_ridge_pipeline


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