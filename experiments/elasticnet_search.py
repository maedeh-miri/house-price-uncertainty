"""Compare ElasticNet hyperparameters using training-only CV."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from house_price_uncertainty.data import load_ames_housing
from house_price_uncertainty.feature_schema import prepare_primary_model_data
from house_price_uncertainty.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from house_price_uncertainty.model_selection import make_training_cv
from house_price_uncertainty.models import build_elasticnet_pipeline
from house_price_uncertainty.splitting import make_random_evaluation_split

RESULTS_DIRECTORY = Path("experiments/results")
RESULTS_PATH = RESULTS_DIRECTORY / "elasticnet_search.csv"
SUMMARY_PATH = RESULTS_DIRECTORY / "elasticnet_search_summary.json"

ELASTIC_NET_ALPHAS = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
)

ELASTIC_NET_L1_RATIOS = (
    0.1,
    0.5,
    0.9,
)


def evaluate_candidate(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    alpha: float,
    l1_ratio: float,
) -> dict[str, float | int]:
    """Evaluate one ElasticNet candidate using frozen training-only CV."""
    cv = make_training_cv()

    out_of_fold_predictions = np.empty(
        len(target),
        dtype=float,
    )

    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    fold_iterations: list[int] = []

    convergence_warnings = 0

    for fit_indices, validation_indices in cv.split(features):
        x_fit = features.iloc[fit_indices]
        y_fit = target.iloc[fit_indices]

        x_validation = features.iloc[validation_indices]
        y_validation = target.iloc[validation_indices]

        pipeline = build_elasticnet_pipeline(
            alpha=alpha,
            l1_ratio=l1_ratio,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            pipeline.fit(
                x_fit,
                y_fit,
            )

        convergence_warnings += sum(
            issubclass(
                warning.category,
                ConvergenceWarning,
            )
            for warning in caught_warnings
        )

        model = pipeline.named_steps["model"]
        fold_iterations.append(int(model.n_iter_))

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
        "l1_ratio": l1_ratio,
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
        "max_iterations": max(fold_iterations),
        "convergence_warnings": convergence_warnings,
    }


def main() -> None:
    """Run the controlled ElasticNet hyperparameter comparison."""
    data = load_ames_housing()

    evaluation_split = make_random_evaluation_split(data)

    training_data = prepare_primary_model_data(
        evaluation_split.train
    )

    features = training_data.features.reset_index(drop=True)
    target = training_data.target.reset_index(drop=True)

    results = []

    for alpha in ELASTIC_NET_ALPHAS:
        for l1_ratio in ELASTIC_NET_L1_RATIOS:
            print(
                "Evaluating "
                f"alpha={alpha}, "
                f"l1_ratio={l1_ratio}"
            )

            results.append(
                evaluate_candidate(
                    features=features,
                    target=target,
                    alpha=alpha,
                    l1_ratio=l1_ratio,
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

    summary = {
        "selection_metric": "oof_mae",
        "secondary_metric": "oof_rmse",
        "candidate_alphas": list(ELASTIC_NET_ALPHAS),
        "candidate_l1_ratios": list(ELASTIC_NET_L1_RATIOS),
        "best_alpha": float(best_row["alpha"]),
        "best_l1_ratio": float(best_row["l1_ratio"]),
        "best_oof_mae": float(best_row["oof_mae"]),
        "best_oof_rmse": float(best_row["oof_rmse"]),
        "best_candidate_convergence_warnings": int(
            best_row["convergence_warnings"]
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
    print("ElasticNet search")
    print("-----------------")
    print(
        results_frame[
            [
                "alpha",
                "l1_ratio",
                "oof_mae",
                "oof_rmse",
                "convergence_warnings",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Best parameters by OOF MAE: "
        f"alpha={summary['best_alpha']}, "
        f"l1_ratio={summary['best_l1_ratio']}"
    )
    print(f"Best OOF MAE:  {summary['best_oof_mae']:,.2f}")
    print(f"Best OOF RMSE: {summary['best_oof_rmse']:,.2f}")
    print(
        "Convergence warnings: "
        f"{summary['best_candidate_convergence_warnings']}"
    )


if __name__ == "__main__":
    main()