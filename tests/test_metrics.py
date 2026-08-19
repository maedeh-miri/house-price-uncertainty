"""Tests for point-prediction and prediction-interval metrics."""

from __future__ import annotations

import numpy as np
import pytest

from house_price_uncertainty.metrics import (
    covered_count,
    empirical_coverage,
    interval_widths,
    mean_absolute_error,
    mean_interval_width,
    root_mean_squared_error,
)


def test_point_metrics_match_manual_calculation() -> None:
    y_true = np.array(
        [
            100.0,
            200.0,
            300.0,
        ]
    )
    y_pred = np.array(
        [
            90.0,
            220.0,
            280.0,
        ]
    )

    assert mean_absolute_error(
        y_true,
        y_pred,
    ) == pytest.approx(
        50.0 / 3.0
    )

    assert root_mean_squared_error(
        y_true,
        y_pred,
    ) == pytest.approx(
        np.sqrt(300.0)
    )


def test_empirical_coverage_is_inclusive() -> None:
    y_true = [
        100.0,
        200.0,
        300.0,
        400.0,
    ]
    lower = [
        100.0,
        150.0,
        310.0,
        350.0,
    ]
    upper = [
        120.0,
        200.0,
        330.0,
        390.0,
    ]

    assert empirical_coverage(
        y_true,
        lower,
        upper,
    ) == pytest.approx(0.5)


def test_covered_count_is_inclusive() -> None:
    count = covered_count(
        y_true=[
            100.0,
            200.0,
            300.0,
            400.0,
        ],
        lower=[
            100.0,
            150.0,
            310.0,
            350.0,
        ],
        upper=[
            120.0,
            200.0,
            330.0,
            390.0,
        ],
    )

    assert count == 2


def test_interval_widths() -> None:
    widths = interval_widths(
        lower=[
            90.0,
            180.0,
        ],
        upper=[
            110.0,
            240.0,
        ],
    )

    assert np.array_equal(
        widths,
        np.array(
            [
                20.0,
                60.0,
            ]
        ),
    )


def test_mean_interval_width() -> None:
    assert mean_interval_width(
        [90.0, 180.0],
        [110.0, 240.0],
    ) == pytest.approx(40.0)


def test_zero_width_interval_is_valid() -> None:
    y_true = [100.0]

    assert covered_count(
        y_true,
        [100.0],
        [100.0],
    ) == 1

    assert empirical_coverage(
        y_true,
        [100.0],
        [100.0],
    ) == pytest.approx(1.0)

    assert mean_interval_width(
        [100.0],
        [100.0],
    ) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "metric_name",
    [
        "coverage",
        "width",
    ],
)
def test_rejects_reversed_interval(
    metric_name: str,
) -> None:
    if metric_name == "coverage":
        with pytest.raises(
            ValueError,
            match="lower bound",
        ):
            empirical_coverage(
                [100.0],
                [110.0],
                [90.0],
            )
    else:
        with pytest.raises(
            ValueError,
            match="lower bound",
        ):
            mean_interval_width(
                [110.0],
                [90.0],
            )


def test_point_metric_rejects_mismatched_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        mean_absolute_error(
            [1.0, 2.0],
            [1.0],
        )


def test_interval_metric_rejects_mismatched_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        empirical_coverage(
            y_true=[1.0, 2.0],
            lower=[0.0],
            upper=[3.0],
        )


def test_point_metric_rejects_nonfinite_values() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        root_mean_squared_error(
            [1.0, np.nan],
            [1.0, 2.0],
        )


def test_interval_metric_rejects_nonfinite_values() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        empirical_coverage(
            y_true=[1.0, 2.0],
            lower=[0.0, np.nan],
            upper=[2.0, 3.0],
        )