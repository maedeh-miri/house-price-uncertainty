"""Benchmark strategies for imputing missing Lot Frontage values.

The experiment compares simple median baselines with a model-based
candidate under two evaluation protocols:

1. repeated random cross-validation on observed frontage values
2. conditional masking weighted by the dataset's observed missingness pattern

All imputation statistics and model parameters are learned from each
training split only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from house_price_uncertainty.data import load_ames_housing

TARGET = "Lot Frontage"
GROUP_COLUMNS = ["Neighborhood", "Lot Config"]

REPEATED_CV_SPLITS = 5
REPEATED_CV_REPEATS = 10
CONDITIONAL_MASKING_REPEATS = 20
RANDOM_SEED = 42
PROPENSITY_SMOOTHING = 10.0

MEDIAN_STRATEGIES = [
    "global_median",
    "neighborhood_median",
    "lot_config_median",
    "hierarchical_median",
]
ALL_STRATEGIES = MEDIAN_STRATEGIES + ["model_based_hgb"]

MODEL_NUMERIC_FEATURES = [
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "1st Flr SF",
    "Gr Liv Area",
]

MODEL_CATEGORICAL_FEATURES = [
    "Neighborhood",
    "Lot Config",
    "Lot Shape",
    "Land Contour",
    "MS Zoning",
    "Street",
    "Alley",
    "Land Slope",
    "Bldg Type",
    "House Style",
]

MODEL_FEATURES = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES

OUTPUT_DIRECTORY = Path("experiments/results")
DETAILED_RESULTS_PATH = OUTPUT_DIRECTORY / "lot_frontage_imputation_detailed.csv"
SUMMARY_RESULTS_PATH = OUTPUT_DIRECTORY / "lot_frontage_imputation_summary.csv"
METADATA_PATH = OUTPUT_DIRECTORY / "lot_frontage_imputation_metadata.json"


def group_median_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_columns: list[str],
) -> pd.Series:
    """Predict target values using group medians learned from training data."""
    statistics = (
        train.groupby(
            group_columns,
            observed=True,
            dropna=False,
        )[TARGET]
        .median()
        .rename("prediction")
        .reset_index()
    )

    test_keys = test[group_columns].copy()
    test_keys["_row_order"] = np.arange(len(test_keys))

    predictions = (
        test_keys.merge(
            statistics,
            on=group_columns,
            how="left",
            validate="many_to_one",
            sort=False,
        )
        .sort_values("_row_order")["prediction"]
        .reset_index(drop=True)
    )

    return predictions.astype(float)


def predict_with_median_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    strategy: str,
) -> tuple[pd.Series, float]:
    """Predict frontage using one leakage-safe median strategy."""
    global_median = float(train[TARGET].median())

    if strategy == "global_median":
        predictions = pd.Series(
            global_median,
            index=range(len(test)),
            dtype=float,
        )
        return predictions, 0.0

    if strategy == "neighborhood_median":
        primary = group_median_prediction(
            train,
            test,
            ["Neighborhood"],
        )
        fallback_rate = float(primary.isna().mean())
        return primary.fillna(global_median), fallback_rate

    if strategy == "lot_config_median":
        primary = group_median_prediction(
            train,
            test,
            ["Lot Config"],
        )
        fallback_rate = float(primary.isna().mean())
        return primary.fillna(global_median), fallback_rate

    if strategy == "hierarchical_median":
        primary = group_median_prediction(
            train,
            test,
            ["Neighborhood", "Lot Config"],
        )
        neighborhood_fallback = group_median_prediction(
            train,
            test,
            ["Neighborhood"],
        )
        fallback_rate = float(primary.isna().mean())

        predictions = (
            primary
            .fillna(neighborhood_fallback)
            .fillna(global_median)
        )
        return predictions, fallback_rate

    raise ValueError(f"Unknown median strategy: {strategy}")


def build_model_based_imputer(random_state: int) -> Pipeline:
    """Build a multivariate frontage-regression pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                MODEL_NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                MODEL_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regressor", regressor),
        ]
    )


def predict_with_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    random_state: int,
) -> pd.Series:
    """Fit the model-based imputer on one training split and predict the test split."""
    model = build_model_based_imputer(random_state)

    model.fit(
        train[MODEL_FEATURES],
        train[TARGET],
    )

    predictions = model.predict(test[MODEL_FEATURES])
    predictions = np.clip(predictions, a_min=0.0, a_max=None)

    return pd.Series(predictions, dtype=float)


def score_predictions(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[float, float]:
    """Return MAE and RMSE."""
    return (
        float(mean_absolute_error(actual, predicted)),
        float(root_mean_squared_error(actual, predicted)),
    )


def evaluate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    evaluation: str,
    split_id: int,
    repeat: int,
    fold: int | None,
    random_state: int,
) -> list[dict[str, object]]:
    """Evaluate all candidate strategies on one train/test split."""
    actual = test[TARGET].reset_index(drop=True)
    records: list[dict[str, object]] = []

    for strategy in MEDIAN_STRATEGIES:
        predicted, fallback_rate = predict_with_median_strategy(
            train,
            test,
            strategy,
        )
        mae, rmse = score_predictions(actual, predicted)

        records.append(
            {
                "evaluation": evaluation,
                "split_id": split_id,
                "repeat": repeat,
                "fold": fold,
                "strategy": strategy,
                "test_rows": len(test),
                "mae": mae,
                "rmse": rmse,
                "primary_fallback_percentage": fallback_rate * 100,
            }
        )

    model_predictions = predict_with_model(
        train,
        test,
        random_state=random_state,
    )
    mae, rmse = score_predictions(actual, model_predictions)

    records.append(
        {
            "evaluation": evaluation,
            "split_id": split_id,
            "repeat": repeat,
            "fold": fold,
            "strategy": "model_based_hgb",
            "test_rows": len(test),
            "mae": mae,
            "rmse": rmse,
            "primary_fallback_percentage": np.nan,
        }
    )

    return records


def run_repeated_cross_validation(
    observed: pd.DataFrame,
) -> list[dict[str, object]]:
    """Run 5-fold cross-validation repeated ten times."""
    cross_validation = RepeatedKFold(
        n_splits=REPEATED_CV_SPLITS,
        n_repeats=REPEATED_CV_REPEATS,
        random_state=RANDOM_SEED,
    )

    records: list[dict[str, object]] = []

    for zero_based_split, (train_indices, test_indices) in enumerate(
        cross_validation.split(observed),
    ):
        split_id = zero_based_split + 1
        repeat = zero_based_split // REPEATED_CV_SPLITS + 1
        fold = zero_based_split % REPEATED_CV_SPLITS + 1

        train = observed.iloc[train_indices].reset_index(drop=True)
        test = observed.iloc[test_indices].reset_index(drop=True)

        records.extend(
            evaluate_split(
                train,
                test,
                evaluation="repeated_random_cv",
                split_id=split_id,
                repeat=repeat,
                fold=fold,
                random_state=RANDOM_SEED + split_id,
            )
        )

    return records


def estimate_missingness_propensity(
    data: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.Series:
    """Estimate smoothed missingness propensity by neighborhood and lot config."""
    global_missing_rate = float(data[TARGET].isna().mean())

    group_statistics = (
        data.assign(_target_missing=data[TARGET].isna())
        .groupby(
            GROUP_COLUMNS,
            observed=True,
            dropna=False,
        )
        .agg(
            rows=("_target_missing", "size"),
            missing_count=("_target_missing", "sum"),
        )
        .reset_index()
    )

    group_statistics["propensity"] = (
        group_statistics["missing_count"]
        + PROPENSITY_SMOOTHING * global_missing_rate
    ) / (
        group_statistics["rows"]
        + PROPENSITY_SMOOTHING
    )

    propensities = (
        observed[GROUP_COLUMNS]
        .merge(
            group_statistics[GROUP_COLUMNS + ["propensity"]],
            on=GROUP_COLUMNS,
            how="left",
            validate="many_to_one",
            sort=False,
        )["propensity"]
        .astype(float)
    )

    return propensities.fillna(global_missing_rate)


def run_conditional_masking(
    data: pd.DataFrame,
    observed: pd.DataFrame,
) -> list[dict[str, object]]:
    """Mask observed values using weights that mimic natural missingness patterns."""
    propensities = estimate_missingness_propensity(data, observed)
    weights = propensities.to_numpy(dtype=float)
    weights = weights / weights.sum()

    natural_missing_rate = float(data[TARGET].isna().mean())
    mask_size = round(natural_missing_rate * len(observed))

    records: list[dict[str, object]] = []
    all_indices = np.arange(len(observed))

    for repeat in range(1, CONDITIONAL_MASKING_REPEATS + 1):
        random_state = RANDOM_SEED + 10_000 + repeat
        random_generator = np.random.default_rng(random_state)

        test_indices = random_generator.choice(
            all_indices,
            size=mask_size,
            replace=False,
            p=weights,
        )

        test_mask = np.zeros(len(observed), dtype=bool)
        test_mask[test_indices] = True

        train = observed.loc[~test_mask].reset_index(drop=True)
        test = observed.loc[test_mask].reset_index(drop=True)

        records.extend(
            evaluate_split(
                train,
                test,
                evaluation="conditional_masking",
                split_id=repeat,
                repeat=repeat,
                fold=None,
                random_state=random_state,
            )
        )

    return records


def summarize_results(
    detailed: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact summary with stability and win-rate diagnostics."""
    summary = (
        detailed.groupby(
            ["evaluation", "strategy"],
            observed=True,
        )
        .agg(
            evaluations=("split_id", "nunique"),
            mean_test_rows=("test_rows", "mean"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            median_mae=("mae", "median"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_primary_fallback_percentage=(
                "primary_fallback_percentage",
                "mean",
            ),
        )
        .reset_index()
    )

    best_indices = detailed.groupby(
        ["evaluation", "split_id"],
        observed=True,
    )["mae"].idxmin()

    wins = (
        detailed.loc[best_indices]
        .groupby(
            ["evaluation", "strategy"],
            observed=True,
        )
        .size()
        .rename("wins")
        .reset_index()
    )

    evaluation_counts = (
        detailed.groupby("evaluation", observed=True)["split_id"]
        .nunique()
        .rename("total_evaluations")
        .reset_index()
    )

    summary = summary.merge(
        wins,
        on=["evaluation", "strategy"],
        how="left",
    )
    summary = summary.merge(
        evaluation_counts,
        on="evaluation",
        how="left",
    )

    summary["wins"] = summary["wins"].fillna(0).astype(int)
    summary["mae_win_percentage"] = (
        summary["wins"]
        .div(summary["total_evaluations"])
        .mul(100)
    )

    global_scores = (
        detailed.loc[
            detailed["strategy"].eq("global_median"),
            ["evaluation", "split_id", "mae", "rmse"],
        ]
        .rename(
            columns={
                "mae": "global_mae",
                "rmse": "global_rmse",
            }
        )
    )

    comparisons = detailed.merge(
        global_scores,
        on=["evaluation", "split_id"],
        how="left",
        validate="many_to_one",
    )

    comparisons["mae_improvement_vs_global_percentage"] = (
        comparisons["global_mae"] - comparisons["mae"]
    ).div(comparisons["global_mae"]).mul(100)

    comparisons["rmse_improvement_vs_global_percentage"] = (
        comparisons["global_rmse"] - comparisons["rmse"]
    ).div(comparisons["global_rmse"]).mul(100)

    improvements = (
        comparisons.groupby(
            ["evaluation", "strategy"],
            observed=True,
        )
        .agg(
            mean_mae_improvement_vs_global_percentage=(
                "mae_improvement_vs_global_percentage",
                "mean",
            ),
            mean_rmse_improvement_vs_global_percentage=(
                "rmse_improvement_vs_global_percentage",
                "mean",
            ),
        )
        .reset_index()
    )

    summary = summary.merge(
        improvements,
        on=["evaluation", "strategy"],
        how="left",
        validate="one_to_one",
    )

    strategy_order = {
        strategy: index
        for index, strategy in enumerate(ALL_STRATEGIES)
    }

    summary["_strategy_order"] = summary["strategy"].map(strategy_order)

    return (
        summary.sort_values(
            ["evaluation", "mean_mae", "_strategy_order"]
        )
        .drop(columns="_strategy_order")
        .reset_index(drop=True)
    )


def validate_experiment_inputs(data: pd.DataFrame) -> None:
    """Fail early when required columns or assumptions are missing."""
    required_columns = set(
        [TARGET]
        + GROUP_COLUMNS
        + MODEL_FEATURES
    )
    missing_columns = sorted(required_columns.difference(data.columns))

    if missing_columns:
        names = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing experiment columns: {names}.")

    if "SalePrice" in MODEL_FEATURES:
        raise ValueError(
            "SalePrice must not be used to impute Lot Frontage."
        )

    if data[TARGET].notna().sum() < 100:
        raise ValueError(
            "Too few observed Lot Frontage values for this benchmark."
        )


def save_outputs(
    detailed: pd.DataFrame,
    summary: pd.DataFrame,
    data: pd.DataFrame,
    observed: pd.DataFrame,
) -> None:
    """Persist detailed results, summaries, and experiment metadata."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    detailed.to_csv(DETAILED_RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_RESULTS_PATH, index=False)

    metadata = {
        "target": TARGET,
        "dataset_rows": len(data),
        "observed_target_rows": len(observed),
        "naturally_missing_target_rows": int(data[TARGET].isna().sum()),
        "natural_missing_percentage": float(
            data[TARGET].isna().mean() * 100
        ),
        "repeated_cv": {
            "splits": REPEATED_CV_SPLITS,
            "repeats": REPEATED_CV_REPEATS,
            "total_evaluations": (
                REPEATED_CV_SPLITS * REPEATED_CV_REPEATS
            ),
        },
        "conditional_masking": {
            "repeats": CONDITIONAL_MASKING_REPEATS,
            "propensity_group_columns": GROUP_COLUMNS,
            "propensity_smoothing": PROPENSITY_SMOOTHING,
            "note": (
                "Groups with no observed frontage values cannot be "
                "directly evaluated."
            ),
        },
        "model_features": MODEL_FEATURES,
        "strategies": ALL_STRATEGIES,
        "random_seed": RANDOM_SEED,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete benchmark and save reproducible outputs."""
    data = load_ames_housing()
    validate_experiment_inputs(data)

    observed = (
        data.loc[data[TARGET].notna()]
        .reset_index(drop=True)
    )

    print("Running repeated random cross-validation...")
    repeated_cv_records = run_repeated_cross_validation(observed)

    print("Running conditional masking evaluation...")
    conditional_masking_records = run_conditional_masking(
        data,
        observed,
    )

    detailed = pd.DataFrame(
        repeated_cv_records + conditional_masking_records
    )

    summary = summarize_results(detailed)
    save_outputs(detailed, summary, data, observed)

    display_columns = [
        "evaluation",
        "strategy",
        "mean_mae",
        "std_mae",
        "mean_rmse",
        "std_rmse",
        "mae_win_percentage",
        "mean_mae_improvement_vs_global_percentage",
        "mean_primary_fallback_percentage",
    ]

    print("\n=== LOT FRONTAGE IMPUTATION SUMMARY ===")
    print(
        summary[display_columns]
        .round(3)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(f"- {DETAILED_RESULTS_PATH}")
    print(f"- {SUMMARY_RESULTS_PATH}")
    print(f"- {METADATA_PATH}")


if __name__ == "__main__":
    main()
