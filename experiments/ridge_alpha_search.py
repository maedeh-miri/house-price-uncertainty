"""Compare Ridge regularization strengths using training-only CV."""

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
RESULTS_PATH = RESULTS_DIRECTORY / "ridge_alpha_search.csv"
SUMMARY_PATH = RESULTS_DIRECTORY / "ridge_alpha_search_summary.json"

RIDGE_ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
)


def evaluate_alpha(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    alpha: float,
) -> dict[str, float]:
    """Evaluate one Ridge alpha with frozen training-only CV."""
    cv = make_training_cv()

    out_of_fold_predictions = np.empty(
        len(target),
        dtype=float,
    )

    fold_mae: list[float] = []
    fold_rmse: list[float] = []

    for fit_indices, validation_indices in cv.split(features):
        x_fit = features.iloc[fit_indices]
        y_fit = target.iloc[fit_indices]

        x_validation = features.iloc[validation_indices]
        y_validation = target.iloc[validation_indices]

        pipeline = build_ridge_pipeline(
            alpha=alpha,
        )

        pipeline.fit(
            x_fit,
            y_fit,
        )

        predictions = pipeline.predict(
            x_validation,
        )

        out_of_fold_predictions[validation_indices] = predictions

        fold_mae.append(
            mean_absolute_error(
                y_validation,
                predictions,
            )
        )
        fold_rmse.append(
            root_mean_squared_error(
                y_validation,
                predictions,
            )
        )

    return {
        "alpha": alpha,
        "oof_mae": mean_absolute_error(
            target,
            out_of_fold_predictions,
        ),
        "oof_rmse": root_mean_squared_error(
            target,
            out_of_fold_predictions,
        ),
        "mean_fold_mae": float(np.mean(fold_mae)),
        "std_fold_mae": float(
            np.std(
                fold_mae,
                ddof=1,
            )
        ),
        "mean_fold_rmse": float(np.mean(fold_rmse)),
        "std_fold_rmse": float(
            np.std(
                fold_rmse,
                ddof=1,
            )
        ),
    }


def main() -> None:
    """Run the controlled Ridge alpha comparison."""
    data = load_ames_housing()

    evaluation_split = make_random_evaluation_split(data)

    training_data = prepare_primary_model_data(
        evaluation_split.train
    )

    features = training_data.features.reset_index(drop=True)
    target = training_data.target.reset_index(drop=True)

    results = [
        evaluate_alpha(
            features=features,
            target=target,
            alpha=alpha,
        )
        for alpha in RIDGE_ALPHAS
    ]

    results_frame = pd.DataFrame(results).sort_values(
        "oof_mae",
        ascending=True,
    )

    best_row = results_frame.iloc[0]

    summary = {
        "selection_metric": "oof_mae",
        "candidate_alphas": list(RIDGE_ALPHAS),
        "best_alpha": float(best_row["alpha"]),
        "best_oof_mae": float(best_row["oof_mae"]),
        "best_oof_rmse": float(best_row["oof_rmse"]),
    }

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_frame.to_csv(
        RESULTS_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Ridge alpha search")
    print("------------------")
    print(
        results_frame[
            [
                "alpha",
                "oof_mae",
                "oof_rmse",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Best alpha by OOF MAE: {summary['best_alpha']}")
    print(f"Best OOF MAE:  {summary['best_oof_mae']:,.2f}")
    print(f"Best OOF RMSE: {summary['best_oof_rmse']:,.2f}")


if __name__ == "__main__":
    main()