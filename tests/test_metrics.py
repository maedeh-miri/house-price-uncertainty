import numpy as np
import pytest

from house_price_uncertainty.metrics import (
    empirical_coverage,
    mean_absolute_error,
    mean_interval_width,
    root_mean_squared_error,
)


def test_point_metrics_match_manual_calculation() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([90.0, 220.0, 280.0])

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(50.0 / 3.0)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(300.0))


def test_empirical_coverage_is_inclusive() -> None:
    y_true = [100.0, 200.0, 300.0, 400.0]
    lower = [100.0, 150.0, 310.0, 350.0]
    upper = [120.0, 200.0, 330.0, 390.0]

    assert empirical_coverage(y_true, lower, upper) == pytest.approx(0.5)


def test_mean_interval_width() -> None:
    assert mean_interval_width([90.0, 180.0], [110.0, 240.0]) == pytest.approx(40.0)


def test_rejects_reversed_interval() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        empirical_coverage([100.0], [110.0], [90.0])


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        mean_absolute_error([1.0, 2.0], [1.0])


def test_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        root_mean_squared_error([1.0, np.nan], [1.0, 2.0])
