"""Metrics for point predictions and prediction intervals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import as_1d_float_array, require_same_length


def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Compute mean absolute error with explicit input validation."""
    true = as_1d_float_array(
        y_true,
        name="y_true",
    )
    pred = as_1d_float_array(
        y_pred,
        name="y_pred",
    )

    require_same_length(
        y_true=true,
        y_pred=pred,
    )

    return float(
        np.mean(
            np.abs(true - pred)
        )
    )


def root_mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Compute root mean squared error."""
    true = as_1d_float_array(
        y_true,
        name="y_true",
    )
    pred = as_1d_float_array(
        y_pred,
        name="y_pred",
    )

    require_same_length(
        y_true=true,
        y_pred=pred,
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(true - pred)
            )
        )
    )


def _validated_interval_arrays(
    lower: ArrayLike,
    upper: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return validated lower and upper interval bounds."""
    low = as_1d_float_array(
        lower,
        name="lower",
    )
    high = as_1d_float_array(
        upper,
        name="upper",
    )

    require_same_length(
        lower=low,
        upper=high,
    )

    if np.any(low > high):
        raise ValueError(
            "Every lower bound must be less than or equal to "
            "its upper bound."
        )

    return low, high


def _coverage_mask(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> NDArray[np.bool_]:
    """Return whether each target lies inside its closed interval."""
    true = as_1d_float_array(
        y_true,
        name="y_true",
    )
    low, high = _validated_interval_arrays(
        lower,
        upper,
    )

    require_same_length(
        y_true=true,
        lower=low,
        upper=high,
    )

    return (true >= low) & (true <= high)


def covered_count(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> int:
    """Return the number of targets inside closed prediction intervals."""
    mask = _coverage_mask(
        y_true,
        lower,
        upper,
    )

    return int(
        np.count_nonzero(mask)
    )


def empirical_coverage(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> float:
    """Return the fraction of targets inside closed prediction intervals."""
    mask = _coverage_mask(
        y_true,
        lower,
        upper,
    )

    return float(
        np.mean(mask)
    )


def interval_widths(
    lower: ArrayLike,
    upper: ArrayLike,
) -> NDArray[np.float64]:
    """Return the width of each prediction interval."""
    low, high = _validated_interval_arrays(
        lower,
        upper,
    )

    return high - low


def mean_interval_width(
    lower: ArrayLike,
    upper: ArrayLike,
) -> float:
    """Return the mean width of prediction intervals."""
    widths = interval_widths(
        lower,
        upper,
    )

    return float(
        np.mean(widths)
    )