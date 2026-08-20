"""Experiment 009: temporal stress test for the frozen primary pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from house_price_uncertainty.conformal import (
    calibrate_symmetric_conformal,
    symmetric_prediction_interval,
)
from house_price_uncertainty.data import load_ames_housing
from house_price_uncertainty.feature_schema import prepare_primary_model_data
from house_price_uncertainty.metrics import (
    empirical_coverage,
    mean_absolute_error,
    mean_interval_width,
    root_mean_squared_error,
)
from house_price_uncertainty.models import build_elasticnet_pipeline

OUTPUT = Path(
    "experiments/results/temporal_stress_summary.json"
)


def temporal_split(data):
    """Split Ames data into historical train/calibration/future test."""
    year = data["Yr Sold"]

    train = data[year <= 2008]
    calibration = data[year == 2009]
    test = data[year == 2010]

    return train, calibration, test


def main() -> None:
    data = load_ames_housing()

    train_df, calibration_df, test_df = temporal_split(data)

    train = prepare_primary_model_data(train_df)
    calibration = prepare_primary_model_data(calibration_df)
    test = prepare_primary_model_data(test_df)

    model = build_elasticnet_pipeline(
        alpha=0.1,
        l1_ratio=0.9,
    )

    model.fit(
        train.features,
        train.target,
    )

    calibration_pred = model.predict(
        calibration.features
    )

    calibration_result = calibrate_symmetric_conformal(
        y_true=calibration.target,
        y_pred=calibration_pred,
        coverage=0.90,
    )

    test_pred = model.predict(
        test.features
    )

    lower, upper = symmetric_prediction_interval(
        test_pred,
        calibration=calibration_result,
    )

    summary = {
        "experiment_id": "experiment_009",
        "protocol": "temporal_stress_test",
        "split": {
            "train_years": "2006-2008",
            "calibration_year": 2009,
            "test_year": 2010,
            "train_rows": len(train.target),
            "calibration_rows": len(calibration.target),
            "test_rows": len(test.target),
        },
        "point_model": {
            "name": "elasticnet",
            "alpha": 0.1,
            "l1_ratio": 0.9,
        },
        "conformal": {
            "coverage": calibration_result.coverage,
            "radius": calibration_result.radius,
            "quantile_rank": calibration_result.quantile_rank,
            "interval_width": float(
                2 * calibration_result.radius
            ),
        },
        "metrics": {
            "mae": mean_absolute_error(
                test.target,
                test_pred,
            ),
            "rmse": root_mean_squared_error(
                test.target,
                test_pred,
            ),
            "empirical_coverage": empirical_coverage(
                test.target,
                lower,
                upper,
            ),
            "mean_interval_width": mean_interval_width(
                lower,
                upper,
            ),
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Temporal stress test")
    print("--------------------")
    print(f"Train rows: {len(train.target)}")
    print(
        f"Calibration rows: {len(calibration.target)}"
    )
    print(f"Test rows: {len(test.target)}")
    print(
        f"MAE: ${summary['metrics']['mae']:,.2f}"
    )
    print(
        f"RMSE: ${summary['metrics']['rmse']:,.2f}"
    )
    print(
        "Coverage: "
        f"{summary['metrics']['empirical_coverage']:.2%}"
    )
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()