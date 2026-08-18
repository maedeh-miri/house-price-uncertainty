"""Evaluate the Ridge baseline using training-only cross-validation."""

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
from house_price_uncertainty.models import build_ridge_pipeline
from house_price_uncertainty.splitting import make_random_evaluation_split

RESULTS_DIRECTORY = Path("experiments/results")
FOLD_RESULTS_PATH = RESULTS_DIRECTORY / "ridge_baseline_cv_folds.csv"
SUMMARY_PATH = RESULTS_DIRECTORY / "ridge_baseline_cv_summary.json"

RIDGE_ALPHA = 1.0


def main() -> None:
    """Run the training-only Ridge baseline experiment."""
    data = load_ames_housing()

    evaluation_split = make_random_evaluation_split(data)

    training_data = prepare_primary_model_data(
        evaluation_split.train
    )

    features = training_data.features.reset_index(drop=True)
    target = training_data.target.reset_index(drop=True)

    cv = make_training_cv()

    out_of_fold_predictions = np.empty(
        len(target),
        dtype=float,
    )

    fold_results: list[dict[str, float | int]] = []

    for fold_number, (fit_indices, validation_indices) in enumerate(
        cv.split(features),
        start=1,
    ):
        x_fit = features.iloc[fit_indices]
        y_fit = target.iloc[fit_indices]

        x_validation = features.iloc[validation_indices]
        y_validation = target.iloc[validation_indices]

        pipeline = build_ridge_pipeline(
            alpha=RIDGE_ALPHA,
        )

        pipeline.fit(
            x_fit,
            y_fit,
        )

        predictions = pipeline.predict(
            x_validation,
        )

        out_of_fold_predictions[validation_indices] = predictions

        fold_results.append(
            {
                "fold": fold_number,
                "fit_rows": len(fit_indices),
                "validation_rows": len(validation_indices),
                "alpha": RIDGE_ALPHA,
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
        "model": "ridge",
        "alpha": RIDGE_ALPHA,
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

    print("Ridge baseline")
    print("--------------")
    print(f"Alpha: {RIDGE_ALPHA}")
    print(f"Outer training rows: {summary['outer_training_rows']}")
    print(f"OOF MAE:  {summary['oof_mae']:,.2f}")
    print(f"OOF RMSE: {summary['oof_rmse']:,.2f}")
    print()
    print(f"Saved fold results to: {FOLD_RESULTS_PATH}")
    print(f"Saved summary to:      {SUMMARY_PATH}")


if __name__ == "__main__":
    main()