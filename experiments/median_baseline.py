"""Evaluate the median SalePrice baseline using training-only cross-validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from house_price_uncertainty.data import load_ames_housing
from house_price_uncertainty.feature_schema import prepare_primary_model_data
from house_price_uncertainty.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from house_price_uncertainty.model_selection import make_training_cv
from house_price_uncertainty.splitting import make_random_evaluation_split

RESULTS_DIRECTORY = Path("experiments/results")
FOLD_RESULTS_PATH = RESULTS_DIRECTORY / "median_baseline_cv_folds.csv"
SUMMARY_PATH = RESULTS_DIRECTORY / "median_baseline_cv_summary.json"


def main() -> None:
    """Run the training-only median-baseline experiment."""
    data = load_ames_housing()

    evaluation_split = make_random_evaluation_split(data)

    training_data = prepare_primary_model_data(
        evaluation_split.train
    )

    target = training_data.target.reset_index(drop=True)

    cv = make_training_cv()

    out_of_fold_predictions = np.empty(
        len(target),
        dtype=float,
    )

    fold_results: list[dict[str, float | int]] = []

    for fold_number, (fit_indices, validation_indices) in enumerate(
        cv.split(target),
        start=1,
    ):
        y_fit = target.iloc[fit_indices]
        y_validation = target.iloc[validation_indices]

        training_median = float(y_fit.median())

        predictions = np.full(
            len(validation_indices),
            training_median,
            dtype=float,
        )

        out_of_fold_predictions[validation_indices] = predictions

        fold_results.append(
            {
                "fold": fold_number,
                "fit_rows": len(fit_indices),
                "validation_rows": len(validation_indices),
                "training_median": training_median,
                "mae": mean_absolute_error(
                    y_validation,
                    predictions,
                ),
                "rmse": root_mean_squared_error(
                    y_validation,
                    predictions,
                ),
            }
        )

    fold_frame = pd.DataFrame(fold_results)

    summary = {
        "protocol": "training-only 5-fold cross-validation",
        "outer_training_rows": len(target),
        "folds": len(fold_frame),
        "oof_mae": mean_absolute_error(
            target,
            out_of_fold_predictions,
        ),
        "oof_rmse": root_mean_squared_error(
            target,
            out_of_fold_predictions,
        ),
        "mean_fold_mae": float(fold_frame["mae"].mean()),
        "std_fold_mae": float(fold_frame["mae"].std(ddof=1)),
        "mean_fold_rmse": float(fold_frame["rmse"].mean()),
        "std_fold_rmse": float(fold_frame["rmse"].std(ddof=1)),
    }

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_frame.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Median baseline")
    print("----------------")
    print(f"Outer training rows: {summary['outer_training_rows']}")
    print(f"OOF MAE:  {summary['oof_mae']:,.2f}")
    print(f"OOF RMSE: {summary['oof_rmse']:,.2f}")
    print()
    print(f"Saved fold results to: {FOLD_RESULTS_PATH}")
    print(f"Saved summary to:      {SUMMARY_PATH}")


if __name__ == "__main__":
    main()