"""Tests for symmetric split-conformal utilities."""

from __future__ import annotations

import numpy as np
import pytest

from house_price_uncertainty.conformal import (
    SymmetricConformalCalibration,
    absolute_residual_scores,
    calibrate_symmetric_conformal,
    conformal_quantile_rank,
    split_conformal_quantile,
    symmetric_prediction_interval,
)


def test_absolute_residual_scores() -> None:
    scores = absolute_residual_scores(
        y_true=[100.0, 120.0, 80.0],
        y_pred=[90.0, 125.0, 100.0],
    )

    assert np.array_equal(
        scores,
        np.array(
            [
                10.0,
                5.0,
                20.0,
            ]
        ),
    )


def test_absolute_residual_scores_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        absolute_residual_scores(
            y_true=[1.0, 2.0],
            y_pred=[1.0],
        )


def test_absolute_residual_scores_rejects_nonfinite_values() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        absolute_residual_scores(
            y_true=[1.0, np.nan],
            y_pred=[1.0, 2.0],
        )


def test_conformal_quantile_rank_for_primary_protocol() -> None:
    rank = conformal_quantile_rank(
        586,
        coverage=0.90,
    )

    assert rank == 529


@pytest.mark.parametrize(
    "coverage",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
        np.nan,
        np.inf,
    ],
)
def test_conformal_quantile_rank_rejects_invalid_coverage(
    coverage: float,
) -> None:
    with pytest.raises(ValueError):
        conformal_quantile_rank(
            100,
            coverage=coverage,
        )


@pytest.mark.parametrize(
    "n_calibration",
    [
        1.5,
        True,
        "586",
    ],
)
def test_conformal_quantile_rank_rejects_invalid_sample_size_type(
    n_calibration: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        conformal_quantile_rank(
            n_calibration,  # type: ignore[arg-type]
            coverage=0.90,
        )


@pytest.mark.parametrize(
    "n_calibration",
    [
        0,
        -1,
    ],
)
def test_conformal_quantile_rank_rejects_nonpositive_sample_size(
    n_calibration: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        conformal_quantile_rank(
            n_calibration,
            coverage=0.90,
        )


def test_conformal_quantile_rank_rejects_unattainable_finite_coverage() -> None:
    with pytest.raises(
        ValueError,
        match="too high",
    ):
        conformal_quantile_rank(
            3,
            coverage=0.90,
        )


def test_split_conformal_quantile_handles_unsorted_duplicate_scores() -> None:
    quantile = split_conformal_quantile(
        [
            9.0,
            1.0,
            4.0,
            7.0,
            4.0,
        ],
        coverage=0.50,
    )

    assert quantile == pytest.approx(4.0)


def test_split_conformal_quantile_rejects_negative_scores() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        split_conformal_quantile(
            [1.0, -2.0, 3.0],
            coverage=0.80,
        )


def test_split_conformal_quantile_rejects_empty_scores() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        split_conformal_quantile(
            [],
            coverage=0.80,
        )


def test_split_conformal_quantile_rejects_nonfinite_scores() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        split_conformal_quantile(
            [1.0, np.inf, 3.0],
            coverage=0.80,
        )


def test_calibrate_symmetric_conformal_returns_consistent_metadata() -> None:
    calibration = calibrate_symmetric_conformal(
        y_true=[
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
        ],
        y_pred=[
            9.0,
            18.0,
            27.0,
            44.0,
            55.0,
        ],
        coverage=0.50,
    )

    assert calibration.coverage == pytest.approx(0.50)
    assert calibration.n_calibration == 5
    assert calibration.quantile_rank == 3
    assert calibration.radius == pytest.approx(3.0)


def test_calibration_object_rejects_inconsistent_rank() -> None:
    with pytest.raises(
        ValueError,
        match="inconsistent",
    ):
        SymmetricConformalCalibration(
            coverage=0.80,
            n_calibration=9,
            quantile_rank=7,
            radius=10.0,
        )


@pytest.mark.parametrize(
    "radius",
    [
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_calibration_object_rejects_invalid_radius(
    radius: float,
) -> None:
    with pytest.raises(ValueError):
        SymmetricConformalCalibration(
            coverage=0.80,
            n_calibration=9,
            quantile_rank=8,
            radius=radius,
        )


def test_symmetric_prediction_interval() -> None:
    calibration = SymmetricConformalCalibration(
        coverage=0.80,
        n_calibration=9,
        quantile_rank=8,
        radius=25.0,
    )

    lower, upper = symmetric_prediction_interval(
        [100.0, 200.0],
        calibration=calibration,
    )

    assert np.array_equal(
        lower,
        np.array(
            [
                75.0,
                175.0,
            ]
        ),
    )

    assert np.array_equal(
        upper,
        np.array(
            [
                125.0,
                225.0,
            ]
        ),
    )


def test_symmetric_prediction_interval_allows_zero_radius() -> None:
    calibration = SymmetricConformalCalibration(
        coverage=0.80,
        n_calibration=9,
        quantile_rank=8,
        radius=0.0,
    )

    predictions = np.array(
        [
            100.0,
            200.0,
        ]
    )

    lower, upper = symmetric_prediction_interval(
        predictions,
        calibration=calibration,
    )

    assert np.array_equal(
        lower,
        predictions,
    )
    assert np.array_equal(
        upper,
        predictions,
    )