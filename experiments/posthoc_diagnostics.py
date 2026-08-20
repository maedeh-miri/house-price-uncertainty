"""Run post-hoc diagnostics on the frozen primary test evaluation."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from house_price_uncertainty.conformal import (
    SymmetricConformalCalibration,
    symmetric_prediction_interval,
)
from house_price_uncertainty.data import load_ames_housing
from house_price_uncertainty.feature_schema import (
    prepare_primary_model_data,
)
from house_price_uncertainty.metrics import (
    empirical_coverage,
    mean_absolute_error,
    mean_interval_width,
    root_mean_squared_error,
)
from house_price_uncertainty.models import build_elasticnet_pipeline
from house_price_uncertainty.splitting import (
    make_random_evaluation_split,
)

EXPERIMENT_ID = "experiment_008"
STAGE = "posthoc_primary_test_diagnostics"
PROTOCOL_NAME = "primary_random"

EXPECTED_TRAIN_ROWS = 1_758
EXPECTED_CALIBRATION_ROWS = 586
EXPECTED_TEST_ROWS = 586

POINT_MODEL_NAME = "elasticnet"
ELASTICNET_ALPHA = 0.1
ELASTICNET_L1_RATIO = 0.9

NOMINAL_COVERAGE = 0.90
EXPECTED_QUANTILE_RANK = 529
EXPECTED_RADIUS = 32616.33610273435

PRIMARY_MINIMUM_N = 50
EXPLORATORY_MINIMUM_N = 20

FINAL_SUMMARY_PATH = Path(
    "experiments/results/final_primary_test_summary.json"
)
OUTPUT_PATH = Path(
    "experiments/results/posthoc_diagnostics_summary.json"
)
TOP_ERRORS_PATH = Path(
    "experiments/results/posthoc_top_errors.csv"
)

TAIL_FRACTIONS = (0.01, 0.05, 0.10)
TOP_ERROR_COUNT = 10


def _assert_close(
    actual: float,
    expected: float,
    *,
    label: str,
    atol: float = 1e-8,
) -> None:
    """Require a reproduced numeric value to match the frozen artifact."""
    if not np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=atol,
    ):
        raise RuntimeError(
            f"{label} does not match the frozen final evaluation: "
            f"{actual!r} != {expected!r}"
        )


def _load_frozen_final_summary() -> dict[str, Any]:
    """Load and validate the committed final primary result."""
    raw = json.loads(
        FINAL_SUMMARY_PATH.read_text(encoding="utf-8")
    )

    if raw["experiment_id"] != "experiment_007":
        raise RuntimeError(
            "Unexpected experiment ID in final summary."
        )

    if raw["stage"] != "final_primary_test":
        raise RuntimeError(
            "Unexpected stage in final summary."
        )

    if raw["protocol"] != PROTOCOL_NAME:
        raise RuntimeError(
            "Unexpected protocol in final summary."
        )

    if raw["test_evaluated"] is not True:
        raise RuntimeError(
            "Final summary does not mark Test as evaluated."
        )

    point_model = raw["point_model"]

    if point_model["name"] != POINT_MODEL_NAME:
        raise RuntimeError(
            "Unexpected point model in final summary."
        )

    _assert_close(
        float(point_model["alpha"]),
        ELASTICNET_ALPHA,
        label="ElasticNet alpha",
    )
    _assert_close(
        float(point_model["l1_ratio"]),
        ELASTICNET_L1_RATIO,
        label="ElasticNet l1_ratio",
    )

    if point_model["target"] != "SalePrice":
        raise RuntimeError(
            "Unexpected target in final summary."
        )

    if point_model["target_transform"] != "none":
        raise RuntimeError(
            "Unexpected target transformation."
        )

    data = raw["data"]

    if int(data["train_rows"]) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Unexpected training-row count."
        )

    if int(data["test_rows"]) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Unexpected test-row count."
        )

    conformal = raw["conformal"]

    if int(conformal["calibration_rows"]) != EXPECTED_CALIBRATION_ROWS:
        raise RuntimeError(
            "Unexpected calibration-row count."
        )

    if conformal["method"] != "split_conformal":
        raise RuntimeError(
            "Unexpected conformal method."
        )

    if conformal["score"] != "absolute_residual":
        raise RuntimeError(
            "Unexpected conformal score."
        )

    if conformal["interval_type"] != "symmetric":
        raise RuntimeError(
            "Unexpected conformal interval type."
        )

    if conformal["clip_lower_bound_at_zero"] is not False:
        raise RuntimeError(
            "Unexpected lower-bound clipping policy."
        )

    _assert_close(
        float(conformal["nominal_coverage"]),
        NOMINAL_COVERAGE,
        label="Nominal coverage",
    )

    if int(conformal["quantile_rank"]) != EXPECTED_QUANTILE_RANK:
        raise RuntimeError(
            "Unexpected conformal quantile rank."
        )

    _assert_close(
        float(conformal["radius"]),
        EXPECTED_RADIUS,
        label="Conformal radius",
    )

    return raw


def _subgroup_interpretation(n: int) -> str:
    """Return the frozen Neighborhood interpretation tier."""
    if n >= PRIMARY_MINIMUM_N:
        return "primary"

    if n >= EXPLORATORY_MINIMUM_N:
        return "exploratory"

    return "descriptive_only"


def _residual_metrics(
    *,
    y_true: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float | int]:
    """Return point-error summaries for a non-empty subset."""
    if y_true.size == 0:
        raise ValueError(
            "Residual metrics require at least one observation."
        )

    predictions = y_true - residuals

    return {
        "n": int(y_true.size),
        "mean_signed_residual": float(np.mean(residuals)),
        "mae": mean_absolute_error(
            y_true,
            predictions,
        ),
        "rmse": root_mean_squared_error(
            y_true,
            predictions,
        ),
    }


def _direction_summary(
    *,
    y_true: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, Any]:
    """Summarize underprediction and overprediction behavior."""
    n_total = int(y_true.size)

    masks = {
        "underprediction": residuals > 0,
        "overprediction": residuals < 0,
        "exact_prediction": residuals == 0,
    }

    result: dict[str, Any] = {}

    for label, mask in masks.items():
        n = int(np.count_nonzero(mask))

        entry: dict[str, Any] = {
            "n": n,
            "fraction": n / n_total,
        }

        if n > 0:
            entry.update(
                _residual_metrics(
                    y_true=y_true[mask],
                    residuals=residuals[mask],
                )
            )

        result[label] = entry

    return result


def _training_price_bands(
    train_target: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return frozen training-target quartile boundaries."""
    q25, q50, q75 = np.quantile(
        train_target,
        [0.25, 0.50, 0.75],
    )

    boundaries = np.asarray(
        [q25, q50, q75],
        dtype=float,
    )

    metadata = {
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
    }

    return boundaries, metadata


def _assign_price_bands(
    target: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Assign values to Q1-Q4 using training-derived cut points."""
    band_index = np.digitize(
        target,
        boundaries,
        right=True,
    )

    labels = np.asarray(
        ["Q1", "Q2", "Q3", "Q4"],
        dtype=object,
    )

    return labels[band_index]


def _price_band_summary(
    *,
    labels: np.ndarray,
    y_true: np.ndarray,
    residuals: np.ndarray,
    covered: np.ndarray,
) -> list[dict[str, Any]]:
    """Return point and interval diagnostics by price band."""
    results: list[dict[str, Any]] = []

    for label in ("Q1", "Q2", "Q3", "Q4"):
        mask = labels == label
        band_true = y_true[mask]
        band_residuals = residuals[mask]
        band_covered = covered[mask]

        metrics = _residual_metrics(
            y_true=band_true,
            residuals=band_residuals,
        )

        results.append(
            {
                "band": label,
                "n": int(band_true.size),
                "mean_true_sale_price": float(
                    np.mean(band_true)
                ),
                "mean_signed_residual": metrics[
                    "mean_signed_residual"
                ],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "covered_count": int(
                    np.count_nonzero(band_covered)
                ),
                "empirical_coverage": float(
                    np.mean(band_covered)
                ),
            }
        )

    return results


def _tail_sensitivity(
    *,
    orders: np.ndarray,
    y_true: np.ndarray,
    predictions: np.ndarray,
    absolute_errors: np.ndarray,
) -> dict[str, Any]:
    """Measure RMSE after removing fixed upper-error fractions."""
    ranking = np.lexsort(
        (
            orders,
            -absolute_errors,
        )
    )

    result: dict[str, Any] = {
        "full_test_rmse": root_mean_squared_error(
            y_true,
            predictions,
        ),
        "removal_rule": (
            "ceil(test_rows * fraction), ranked by descending "
            "absolute error with Order ascending as tie-breaker"
        ),
        "fractions": {},
    }

    for fraction in TAIL_FRACTIONS:
        remove_count = math.ceil(
            y_true.size * fraction
        )

        keep = np.ones(
            y_true.size,
            dtype=bool,
        )
        keep[ranking[:remove_count]] = False

        key = f"{int(fraction * 100)}_percent"

        result["fractions"][key] = {
            "fraction_requested": fraction,
            "removed_count": remove_count,
            "remaining_count": int(
                np.count_nonzero(keep)
            ),
            "rmse_after_removal": (
                root_mean_squared_error(
                    y_true[keep],
                    predictions[keep],
                )
            ),
        }

    return result


def _conformal_miss_summary(
    *,
    y_true: np.ndarray,
    predictions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    covered: np.ndarray,
    price_labels: np.ndarray,
) -> dict[str, Any]:
    """Summarize observations outside the frozen intervals."""
    missed = ~covered
    above_upper = y_true > upper
    below_lower = y_true < lower

    misses_by_band = {
        label: int(
            np.count_nonzero(
                missed & (price_labels == label)
            )
        )
        for label in ("Q1", "Q2", "Q3", "Q4")
    }

    return {
        "total_misses": int(
            np.count_nonzero(missed)
        ),
        "misses_above_upper": int(
            np.count_nonzero(above_upper)
        ),
        "misses_below_lower": int(
            np.count_nonzero(below_lower)
        ),
        "covered_observations": {
            "n": int(
                np.count_nonzero(covered)
            ),
            "mae": mean_absolute_error(
                y_true[covered],
                predictions[covered],
            ),
            "rmse": root_mean_squared_error(
                y_true[covered],
                predictions[covered],
            ),
        },
        "missed_observations": {
            "n": int(
                np.count_nonzero(missed)
            ),
            "mae": mean_absolute_error(
                y_true[missed],
                predictions[missed],
            ),
            "rmse": root_mean_squared_error(
                y_true[missed],
                predictions[missed],
            ),
        },
        "misses_by_training_price_band": misses_by_band,
    }


def _neighborhood_summary(
    *,
    neighborhoods: np.ndarray,
    y_true: np.ndarray,
    residuals: np.ndarray,
    covered: np.ndarray,
) -> list[dict[str, Any]]:
    """Return frozen-policy Neighborhood diagnostics."""
    results: list[dict[str, Any]] = []

    for neighborhood in sorted(
        np.unique(neighborhoods).tolist()
    ):
        mask = neighborhoods == neighborhood
        subgroup_true = y_true[mask]
        subgroup_residuals = residuals[mask]
        subgroup_covered = covered[mask]

        metrics = _residual_metrics(
            y_true=subgroup_true,
            residuals=subgroup_residuals,
        )

        n = int(subgroup_true.size)

        results.append(
            {
                "neighborhood": str(neighborhood),
                "n": n,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mean_signed_residual": metrics[
                    "mean_signed_residual"
                ],
                "covered_count": int(
                    np.count_nonzero(
                        subgroup_covered
                    )
                ),
                "empirical_coverage": float(
                    np.mean(subgroup_covered)
                ),
                "interpretation": (
                    _subgroup_interpretation(n)
                ),
            }
        )

    return results


def _top_error_table(
    *,
    orders: np.ndarray,
    neighborhoods: np.ndarray,
    y_true: np.ndarray,
    predictions: np.ndarray,
    residuals: np.ndarray,
    absolute_errors: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    covered: np.ndarray,
) -> pd.DataFrame:
    """Return the fixed top-10 largest absolute errors."""
    table = pd.DataFrame(
        {
            "Order": orders,
            "Neighborhood": neighborhoods,
            "y_true": y_true,
            "y_pred": predictions,
            "residual": residuals,
            "absolute_error": absolute_errors,
            "lower_bound": lower,
            "upper_bound": upper,
            "covered": covered,
        }
    )

    table["miss_direction"] = np.where(
        table["covered"],
        "covered",
        np.where(
            table["y_true"] > table["upper_bound"],
            "above_upper",
            "below_lower",
        ),
    )

    return (
        table.sort_values(
            by=["absolute_error", "Order"],
            ascending=[False, True],
        )
        .head(TOP_ERROR_COUNT)
        .reset_index(drop=True)
    )


def main() -> None:
    """Run the frozen post-hoc primary-test diagnostics."""
    frozen_summary = _load_frozen_final_summary()

    data = load_ames_housing()
    split = make_random_evaluation_split(
        data,
        first_random_state=42,
        second_random_state=43,
    )

    if len(split.train) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Unexpected training partition size."
        )

    if len(split.calibration) != EXPECTED_CALIBRATION_ROWS:
        raise RuntimeError(
            "Unexpected calibration partition size."
        )

    if len(split.test) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Unexpected test partition size."
        )

    train = prepare_primary_model_data(
        split.train
    )
    test = prepare_primary_model_data(
        split.test
    )

    if "Neighborhood" not in test.features.columns:
        raise RuntimeError(
            "Neighborhood is required for diagnostics."
        )

    if "Order" not in split.test.columns:
        raise RuntimeError(
            "Order is required for stable diagnostics."
        )

    model = build_elasticnet_pipeline(
        alpha=ELASTICNET_ALPHA,
        l1_ratio=ELASTICNET_L1_RATIO,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                ConvergenceWarning,
            )
            model.fit(
                train.features,
                train.target,
            )
    except ConvergenceWarning as exc:
        raise RuntimeError(
            "ElasticNet emitted a convergence warning during "
            "the frozen diagnostic refit."
        ) from exc

    predictions = np.asarray(
        model.predict(test.features),
        dtype=float,
    )

    y_true = np.asarray(
        test.target,
        dtype=float,
    )

    train_target = np.asarray(
        train.target,
        dtype=float,
    )

    orders = np.asarray(
        split.test["Order"],
        dtype=int,
    )

    neighborhoods = (
        test.features["Neighborhood"]
        .astype(str)
        .to_numpy()
    )

    calibration = SymmetricConformalCalibration(
        coverage=NOMINAL_COVERAGE,
        n_calibration=EXPECTED_CALIBRATION_ROWS,
        quantile_rank=EXPECTED_QUANTILE_RANK,
        radius=EXPECTED_RADIUS,
    )

    lower, upper = symmetric_prediction_interval(
        predictions,
        calibration=calibration,
    )

    residuals = y_true - predictions
    absolute_errors = np.abs(residuals)
    covered = (
        (y_true >= lower)
        & (y_true <= upper)
    )

    reproduced_mae = mean_absolute_error(
        y_true,
        predictions,
    )
    reproduced_rmse = root_mean_squared_error(
        y_true,
        predictions,
    )
    reproduced_coverage = empirical_coverage(
        y_true,
        lower,
        upper,
    )
    reproduced_width = mean_interval_width(
        lower,
        upper,
    )
    reproduced_covered = int(
        np.count_nonzero(covered)
    )

    _assert_close(
        reproduced_mae,
        float(
            frozen_summary[
                "point_metrics"
            ]["mae"]
        ),
        label="Final Test MAE",
        atol=1e-6,
    )

    _assert_close(
        reproduced_rmse,
        float(
            frozen_summary[
                "point_metrics"
            ]["rmse"]
        ),
        label="Final Test RMSE",
        atol=1e-6,
    )

    _assert_close(
        reproduced_coverage,
        float(
            frozen_summary[
                "interval_metrics"
            ]["empirical_coverage"]
        ),
        label="Final empirical coverage",
        atol=1e-12,
    )

    _assert_close(
        reproduced_width,
        float(
            frozen_summary[
                "interval_metrics"
            ]["mean_interval_width"]
        ),
        label="Final mean interval width",
        atol=1e-6,
    )

    if (
        reproduced_covered
        != int(
            frozen_summary[
                "interval_metrics"
            ]["covered_count"]
        )
    ):
        raise RuntimeError(
            "Covered count does not reproduce the frozen "
            "final evaluation."
        )

    boundaries, boundary_metadata = (
        _training_price_bands(
            train_target
        )
    )

    price_labels = _assign_price_bands(
        y_true,
        boundaries,
    )

    overall = {
        "n": int(y_true.size),
        "mean_residual": float(
            np.mean(residuals)
        ),
        "median_residual": float(
            np.median(residuals)
        ),
        "residual_standard_deviation": float(
            np.std(
                residuals,
                ddof=0,
            )
        ),
        "residual_standard_deviation_ddof": 0,
        "minimum_residual": float(
            np.min(residuals)
        ),
        "maximum_residual": float(
            np.max(residuals)
        ),
        "mae": reproduced_mae,
        "rmse": reproduced_rmse,
        "absolute_error": {
            "median": float(
                np.quantile(
                    absolute_errors,
                    0.50,
                )
            ),
            "p75": float(
                np.quantile(
                    absolute_errors,
                    0.75,
                )
            ),
            "p90": float(
                np.quantile(
                    absolute_errors,
                    0.90,
                )
            ),
            "p95": float(
                np.quantile(
                    absolute_errors,
                    0.95,
                )
            ),
            "p99": float(
                np.quantile(
                    absolute_errors,
                    0.99,
                )
            ),
            "maximum": float(
                np.max(absolute_errors)
            ),
            "quantile_method": "linear",
        },
    }

    tail_sensitivity = _tail_sensitivity(
        orders=orders,
        y_true=y_true,
        predictions=predictions,
        absolute_errors=absolute_errors,
    )

    direction_summary = _direction_summary(
        y_true=y_true,
        residuals=residuals,
    )

    price_band_results = _price_band_summary(
        labels=price_labels,
        y_true=y_true,
        residuals=residuals,
        covered=covered,
    )

    miss_summary = _conformal_miss_summary(
        y_true=y_true,
        predictions=predictions,
        lower=lower,
        upper=upper,
        covered=covered,
        price_labels=price_labels,
    )

    neighborhood_results = _neighborhood_summary(
        neighborhoods=neighborhoods,
        y_true=y_true,
        residuals=residuals,
        covered=covered,
    )

    top_errors = _top_error_table(
        orders=orders,
        neighborhoods=neighborhoods,
        y_true=y_true,
        predictions=predictions,
        residuals=residuals,
        absolute_errors=absolute_errors,
        lower=lower,
        upper=upper,
        covered=covered,
    )

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "analysis_type": "posthoc_diagnostic",
        "protocol": PROTOCOL_NAME,
        "primary_system_frozen": True,
        "model_changes_allowed": False,
        "point_model": {
            "name": POINT_MODEL_NAME,
            "alpha": ELASTICNET_ALPHA,
            "l1_ratio": ELASTICNET_L1_RATIO,
            "target": "SalePrice",
            "target_transform": "none",
        },
        "conformal": {
            "method": "split_conformal",
            "score": "absolute_residual",
            "nominal_coverage": NOMINAL_COVERAGE,
            "calibration_rows": (
                EXPECTED_CALIBRATION_ROWS
            ),
            "quantile_rank": (
                EXPECTED_QUANTILE_RANK
            ),
            "radius": EXPECTED_RADIUS,
            "interval_type": "symmetric",
            "clip_lower_bound_at_zero": False,
        },
        "reproduced_frozen_metrics": {
            "test_rows": int(
                y_true.size
            ),
            "mae": reproduced_mae,
            "rmse": reproduced_rmse,
            "covered_count": reproduced_covered,
            "empirical_coverage": (
                reproduced_coverage
            ),
            "mean_interval_width": (
                reproduced_width
            ),
        },
        "overall_residuals": overall,
        "rmse_tail_sensitivity": (
            tail_sensitivity
        ),
        "direction_summary": (
            direction_summary
        ),
        "price_band_definition": {
            "source": (
                "frozen primary training target"
            ),
            "boundaries": boundary_metadata,
            "assignment_rule": (
                "Q1 <= q25; "
                "Q2 = (q25, q50]; "
                "Q3 = (q50, q75]; "
                "Q4 > q75"
            ),
        },
        "price_band_results": (
            price_band_results
        ),
        "conformal_misses": miss_summary,
        "neighborhood_policy": {
            "primary_minimum_n": (
                PRIMARY_MINIMUM_N
            ),
            "exploratory_minimum_n": (
                EXPLORATORY_MINIMUM_N
            ),
        },
        "neighborhood_results": (
            neighborhood_results
        ),
        "top_error_policy": {
            "count": TOP_ERROR_COUNT,
            "ranking": (
                "absolute_error descending, "
                "Order ascending tie-breaker"
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    top_errors.to_csv(
        TOP_ERRORS_PATH,
        index=False,
    )

    print("Post-hoc primary Test diagnostics")
    print("---------------------------------")
    print(
        f"Test rows: {len(y_true)}"
    )
    print(
        f"Reproduced MAE: "
        f"${reproduced_mae:,.2f}"
    )
    print(
        f"Reproduced RMSE: "
        f"${reproduced_rmse:,.2f}"
    )
    print(
        "Reproduced coverage: "
        f"{reproduced_coverage:.2%}"
    )
    print(
        "Largest absolute error: "
        f"${np.max(absolute_errors):,.2f}"
    )
    print(
        f"Conformal misses: "
        f"{np.count_nonzero(~covered)}"
    )
    print(
        f"Saved summary: {OUTPUT_PATH}"
    )
    print(
        f"Saved top errors: {TOP_ERRORS_PATH}"
    )


if __name__ == "__main__":
    main()