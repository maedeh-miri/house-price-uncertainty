"""Tune Random Forest hyperparameters using training-only CV."""

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
from house_price_uncertainty.models import build_random_forest_pipeline
from house_price_uncertainty.splitting import make_random_evaluation_split

RESULTS_DIRECTORY = Path("experiments/results")
RESULTS_PATH = RESULTS_DIRECTORY / "random_forest_search.csv"
SUMMARY_PATH = RESULTS_DIRECTORY / "random_forest_search_summary.json"

N_ESTIMATORS = 500
RANDOM_STATE = 2026

MAX_DEPTHS = (
    None,
    16,
)

MIN_SAMPLES_LEAVES = (
    1,
    2,
    4,
)

MAX_FEATURES_VALUES = (
    1.0,
    0.7,
)


def evaluate_candidate(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float,
) -> dict[str, float | int | None]:
    """Evaluate one Random Forest candidate using frozen training-only CV."""
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

        pipeline = build_random_forest_pipeline(
            n_estimators=N_ESTIMATORS,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=RANDOM_STATE,
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
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
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
    """Run the controlled Random Forest hyperparameter search."""
    data = load_ames_housing()

    evaluation_split = make_random_evaluation_split(data)

    training_data = prepare_primary_model_data(
        evaluation_split.train
    )

    features = training_data.features.reset_index(drop=True)
    target = training_data.target.reset_index(drop=True)

    results = []

    for max_depth in MAX_DEPTHS:
        for min_samples_leaf in MIN_SAMPLES_LEAVES:
            for max_features in MAX_FEATURES_VALUES:
                print(
                    "Evaluating "
                    f"max_depth={max_depth}, "
                    f"min_samples_leaf={min_samples_leaf}, "
                    f"max_features={max_features}"
                )

                results.append(
                    evaluate_candidate(
                        features=features,
                        target=target,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                    )
                )

    results_frame = pd.DataFrame(results).sort_values(
        [
            "oof_mae",
            "oof_rmse",
        ],
        ascending=True,
    )

    best_row = results_frame.iloc[0]

    best_max_depth = best_row["max_depth"]

    if pd.isna(best_max_depth):
        best_max_depth_value = None
    else:
        best_max_depth_value = int(best_max_depth)

    summary = {
        "selection_metric": "oof_mae",
        "secondary_metric": "oof_rmse",
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "candidate_max_depths": list(MAX_DEPTHS),
        "candidate_min_samples_leaves": list(
            MIN_SAMPLES_LEAVES
        ),
        "candidate_max_features": list(
            MAX_FEATURES_VALUES
        ),
        "best_max_depth": best_max_depth_value,
        "best_min_samples_leaf": int(
            best_row["min_samples_leaf"]
        ),
        "best_max_features": float(
            best_row["max_features"]
        ),
        "best_oof_mae": float(
            best_row["oof_mae"]
        ),
        "best_oof_rmse": float(
            best_row["oof_rmse"]
        ),
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

    print()
    print("Random Forest search")
    print("--------------------")
    print(
        results_frame[
            [
                "max_depth",
                "min_samples_leaf",
                "max_features",
                "oof_mae",
                "oof_rmse",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Best parameters by OOF MAE: "
        f"max_depth={summary['best_max_depth']}, "
        f"min_samples_leaf={summary['best_min_samples_leaf']}, "
        f"max_features={summary['best_max_features']}"
    )
    print(f"Best OOF MAE:  {summary['best_oof_mae']:,.2f}")
    print(f"Best OOF RMSE: {summary['best_oof_rmse']:,.2f}")


if __name__ == "__main__":
    main()