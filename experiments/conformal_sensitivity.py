"""Experiment 010: conformal coverage sensitivity analysis."""

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
    mean_interval_width,
)
from house_price_uncertainty.models import build_elasticnet_pipeline
from house_price_uncertainty.splitting import make_random_evaluation_split

OUTPUT = Path("experiments/results/conformal_sensitivity_summary.json")


def main() -> None:
    data = load_ames_housing()

    split = make_random_evaluation_split(data)

    train = prepare_primary_model_data(split.train)
    calibration = prepare_primary_model_data(split.calibration)
    test = prepare_primary_model_data(split.test)

    model = build_elasticnet_pipeline(
        alpha=0.1,
        l1_ratio=0.9,
    )

    model.fit(train.features, train.target)

    calibration_prediction = model.predict(calibration.features)
    test_prediction = model.predict(test.features)

    results = []

    for coverage in [0.80, 0.90, 0.95]:
        conformal = calibrate_symmetric_conformal(
            y_true=calibration.target,
            y_pred=calibration_prediction,
            coverage=coverage,
        )

        lower, upper = symmetric_prediction_interval(
            y_pred=test_prediction,
            calibration=conformal,
        )

        results.append(
            {
                "nominal_coverage": coverage,
                "radius": conformal.radius,
                "empirical_coverage": empirical_coverage(
                    test.target,
                    lower,
                    upper,
                ),
                "mean_interval_width": mean_interval_width(
                    lower,
                    upper,
                ),
                "covered_count": int(
                    ((test.target >= lower) & (test.target <= upper)).sum()
                ),
                "total_count": len(test.target),
            }
        )

    summary = {
        "experiment_id": "experiment_010",
        "protocol": "conformal_sensitivity_analysis",
        "point_model": {
            "name": "elasticnet",
            "alpha": 0.1,
            "l1_ratio": 0.9,
        },
        "split": {
            "train_rows": len(split.train),
            "calibration_rows": len(split.calibration),
            "test_rows": len(split.test),
        },
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Conformal sensitivity analysis")
    print("------------------------------")

    for result in results:
        print(
            f"{result['nominal_coverage']:.0%} coverage | "
            f"empirical: {result['empirical_coverage']:.2%} | "
            f"width: ${result['mean_interval_width']:,.2f}"
        )

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()