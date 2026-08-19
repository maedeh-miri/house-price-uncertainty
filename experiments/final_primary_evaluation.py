"""Run the frozen final evaluation on the primary test partition."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
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
    covered_count,
    empirical_coverage,
    mean_absolute_error,
    mean_interval_width,
    root_mean_squared_error,
)
from house_price_uncertainty.models import (
    build_elasticnet_pipeline,
)
from house_price_uncertainty.splitting import (
    make_random_evaluation_split,
)

EXPERIMENT_ID = "experiment_007"
STAGE = "final_primary_test"
PROTOCOL_NAME = "primary_random"

EXPECTED_TRAIN_ROWS = 1_758
EXPECTED_CALIBRATION_ROWS = 586
EXPECTED_TEST_ROWS = 586

POINT_MODEL_NAME = "elasticnet"
ELASTICNET_ALPHA = 0.1
ELASTICNET_L1_RATIO = 0.9

NOMINAL_COVERAGE = 0.90
EXPECTED_QUANTILE_RANK = 529

PRIMARY_MINIMUM_N = 50
EXPLORATORY_MINIMUM_N = 20

CALIBRATION_SUMMARY_PATH = Path(
    "experiments/results/conformal_calibration_summary.json"
)

OUTPUT_PATH = Path(
    "experiments/results/final_primary_test_summary.json"
)


def _load_frozen_calibration() -> SymmetricConformalCalibration:
    """Load and validate the committed conformal calibration artifact."""
    raw = json.loads(
        CALIBRATION_SUMMARY_PATH.read_text(
            encoding="utf-8",
        )
    )

    if raw["experiment_id"] != EXPERIMENT_ID:
        raise RuntimeError(
            "Unexpected experiment ID in calibration artifact."
        )

    if raw["protocol"] != PROTOCOL_NAME:
        raise RuntimeError(
            "Unexpected protocol in calibration artifact."
        )

    if raw["test_evaluated"] is not False:
        raise RuntimeError(
            "Calibration artifact indicates that the test partition "
            "was already evaluated."
        )

    point_model = raw["point_model"]

    if point_model["name"] != POINT_MODEL_NAME:
        raise RuntimeError(
            "Unexpected point model in calibration artifact."
        )

    if float(point_model["alpha"]) != ELASTICNET_ALPHA:
        raise RuntimeError(
            "Unexpected ElasticNet alpha in calibration artifact."
        )

    if (
        float(point_model["l1_ratio"])
        != ELASTICNET_L1_RATIO
    ):
        raise RuntimeError(
            "Unexpected ElasticNet l1_ratio in calibration artifact."
        )

    data = raw["data"]

    if int(data["train_rows"]) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Unexpected training-row count in calibration artifact."
        )

    if (
        int(data["calibration_rows"])
        != EXPECTED_CALIBRATION_ROWS
    ):
        raise RuntimeError(
            "Unexpected calibration-row count in calibration artifact."
        )

    conformal = raw["conformal"]

    if float(conformal["nominal_coverage"]) != NOMINAL_COVERAGE:
        raise RuntimeError(
            "Unexpected nominal coverage in calibration artifact."
        )

    if int(conformal["quantile_rank"]) != EXPECTED_QUANTILE_RANK:
        raise RuntimeError(
            "Unexpected quantile rank in calibration artifact."
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

    return SymmetricConformalCalibration(
        coverage=float(
            conformal["nominal_coverage"]
        ),
        n_calibration=int(
            data["calibration_rows"]
        ),
        quantile_rank=int(
            conformal["quantile_rank"]
        ),
        radius=float(
            conformal["radius"]
        ),
    )


def _subgroup_interpretation(
    n: int,
) -> str:
    """Return the frozen interpretation tier for a subgroup."""
    if n >= PRIMARY_MINIMUM_N:
        return "primary"

    if n >= EXPLORATORY_MINIMUM_N:
        return "exploratory"

    return "descriptive_only"


def _neighborhood_results(
    *,
    neighborhoods: np.ndarray,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[dict[str, Any]]:
    """Return frozen Neighborhood-level coverage diagnostics."""
    results: list[dict[str, Any]] = []

    for neighborhood in sorted(
        np.unique(neighborhoods).tolist()
    ):
        mask = neighborhoods == neighborhood

        subgroup_true = y_true[mask]
        subgroup_lower = lower[mask]
        subgroup_upper = upper[mask]

        n = int(
            np.count_nonzero(mask)
        )

        results.append(
            {
                "neighborhood": str(neighborhood),
                "n": n,
                "covered_count": covered_count(
                    subgroup_true,
                    subgroup_lower,
                    subgroup_upper,
                ),
                "empirical_coverage": empirical_coverage(
                    subgroup_true,
                    subgroup_lower,
                    subgroup_upper,
                ),
                "interpretation": (
                    _subgroup_interpretation(n)
                ),
            }
        )

    return results


def main() -> None:
    """Evaluate the frozen point and conformal models once on Test."""
    calibration = _load_frozen_calibration()

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
        split.train,
    )
    test = prepare_primary_model_data(
        split.test,
    )

    if "Neighborhood" not in test.features.columns:
        raise RuntimeError(
            "Neighborhood is required for frozen subgroup evaluation."
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
            "ElasticNet emitted a convergence warning during the "
            "frozen full-training fit."
        ) from exc

    test_predictions = np.asarray(
        model.predict(
            test.features,
        ),
        dtype=float,
    )

    test_target = np.asarray(
        test.target,
        dtype=float,
    )

    lower, upper = symmetric_prediction_interval(
        test_predictions,
        calibration=calibration,
    )

    point_mae = mean_absolute_error(
        test_target,
        test_predictions,
    )
    point_rmse = root_mean_squared_error(
        test_target,
        test_predictions,
    )

    n_covered = covered_count(
        test_target,
        lower,
        upper,
    )
    coverage = empirical_coverage(
        test_target,
        lower,
        upper,
    )
    mean_width = mean_interval_width(
        lower,
        upper,
    )

    neighborhoods = (
        test.features["Neighborhood"]
        .astype(str)
        .to_numpy()
    )

    subgroup_results = _neighborhood_results(
        neighborhoods=neighborhoods,
        y_true=test_target,
        lower=lower,
        upper=upper,
    )

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "protocol": PROTOCOL_NAME,
        "point_model": {
            "name": POINT_MODEL_NAME,
            "alpha": ELASTICNET_ALPHA,
            "l1_ratio": ELASTICNET_L1_RATIO,
            "target": "SalePrice",
            "target_scale": "raw_sale_price",
            "target_transform": "none",
        },
        "data": {
            "train_rows": len(train.target),
            "test_rows": len(test.target),
        },
        "conformal": {
            "method": "split_conformal",
            "score": "absolute_residual",
            "interval_type": "symmetric",
            "nominal_coverage": calibration.coverage,
            "calibration_rows": calibration.n_calibration,
            "quantile_rank": calibration.quantile_rank,
            "radius": calibration.radius,
            "clip_lower_bound_at_zero": False,
        },
        "point_metrics": {
            "mae": point_mae,
            "rmse": point_rmse,
        },
        "interval_metrics": {
            "covered_count": n_covered,
            "total_count": len(test_target),
            "empirical_coverage": coverage,
            "mean_interval_width": mean_width,
        },
        "subgroup_policy": {
            "column": "Neighborhood",
            "primary_minimum_n": PRIMARY_MINIMUM_N,
            "exploratory_minimum_n": EXPLORATORY_MINIMUM_N,
        },
        "subgroup_results": subgroup_results,
        "test_evaluated": True,
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
        )
        + "\n",
        encoding="utf-8",
    )

    print("Final primary Test evaluation")
    print("-----------------------------")
    print(
        f"Test rows: {len(test_target)}"
    )
    print(
        f"MAE: ${point_mae:,.2f}"
    )
    print(
        f"RMSE: ${point_rmse:,.2f}"
    )
    print(
        "Empirical 90% coverage: "
        f"{coverage:.2%}"
    )
    print(
        "Covered observations: "
        f"{n_covered}/{len(test_target)}"
    )
    print(
        "Mean interval width: "
        f"${mean_width:,.2f}"
    )
    print(
        f"Saved summary: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()