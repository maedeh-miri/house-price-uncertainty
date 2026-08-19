"""Utilities for symmetric split-conformal prediction intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import as_1d_float_array, require_same_length


def _validate_coverage(coverage: float) -> float:
    """Validate and normalize a requested nominal coverage level."""
    coverage_value = float(coverage)

    if not math.isfinite(coverage_value):
        raise ValueError("coverage must be finite.")

    if not 0.0 < coverage_value < 1.0:
        raise ValueError(
            "coverage must be strictly between 0 and 1."
        )

    return coverage_value


def conformal_quantile_rank(
    n_calibration: int,
    *,
    coverage: float,
) -> int:
    """Return the one-based finite-sample split-conformal rank."""
    if isinstance(n_calibration, bool) or not isinstance(
        n_calibration,
        Integral,
    ):
        raise TypeError(
            "n_calibration must be an integer."
        )

    n_calibration = int(n_calibration)

    if n_calibration <= 0:
        raise ValueError(
            "n_calibration must be a positive integer."
        )

    coverage_value = _validate_coverage(coverage)

    rank = math.ceil(
        (n_calibration + 1) * coverage_value
    )

    if rank > n_calibration:
        maximum_finite_coverage = (
            n_calibration / (n_calibration + 1)
        )

        raise ValueError(
            "Requested coverage is too high for a finite conformal "
            f"quantile with n_calibration={n_calibration}. "
            "The maximum finite coverage is "
            f"{maximum_finite_coverage:.6f}."
        )

    return rank


@dataclass(frozen=True)
class SymmetricConformalCalibration:
    """Frozen metadata for a symmetric split-conformal calibration."""

    coverage: float
    n_calibration: int
    quantile_rank: int
    radius: float

    def __post_init__(self) -> None:
        """Validate internal consistency of the calibration result."""
        expected_rank = conformal_quantile_rank(
            self.n_calibration,
            coverage=self.coverage,
        )

        if self.quantile_rank != expected_rank:
            raise ValueError(
                "quantile_rank is inconsistent with coverage and "
                "n_calibration."
            )

        if not math.isfinite(self.radius):
            raise ValueError(
                "radius must be finite."
            )

        if self.radius < 0.0:
            raise ValueError(
                "radius must be non-negative."
            )


def absolute_residual_scores(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> NDArray[np.float64]:
    """Return absolute residual nonconformity scores."""
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

    return np.abs(true - pred)


def split_conformal_quantile(
    scores: ArrayLike,
    *,
    coverage: float,
) -> float:
    """Return the finite-sample split-conformal score quantile."""
    score_array = as_1d_float_array(
        scores,
        name="scores",
    )

    if np.any(score_array < 0.0):
        raise ValueError(
            "Conformal nonconformity scores must be non-negative."
        )

    rank = conformal_quantile_rank(
        len(score_array),
        coverage=coverage,
    )

    partitioned = np.partition(
        score_array,
        rank - 1,
    )

    return float(
        partitioned[rank - 1]
    )


def calibrate_symmetric_conformal(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    coverage: float,
) -> SymmetricConformalCalibration:
    """Calibrate a symmetric interval using absolute residual scores."""
    scores = absolute_residual_scores(
        y_true,
        y_pred,
    )

    rank = conformal_quantile_rank(
        len(scores),
        coverage=coverage,
    )

    radius = split_conformal_quantile(
        scores,
        coverage=coverage,
    )

    return SymmetricConformalCalibration(
        coverage=float(coverage),
        n_calibration=len(scores),
        quantile_rank=rank,
        radius=radius,
    )


def symmetric_prediction_interval(
    y_pred: ArrayLike,
    *,
    calibration: SymmetricConformalCalibration,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Construct symmetric intervals from a frozen calibration result."""
    pred = as_1d_float_array(
        y_pred,
        name="y_pred",
    )

    if not isinstance(
        calibration,
        SymmetricConformalCalibration,
    ):
        raise TypeError(
            "calibration must be a "
            "SymmetricConformalCalibration instance."
        )

    lower = pred - calibration.radius
    upper = pred + calibration.radius

    return lower, upper