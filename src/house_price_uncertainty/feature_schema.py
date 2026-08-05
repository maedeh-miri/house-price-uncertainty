"""Feature roles and leakage-safe preparation for the primary model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET_COLUMN = "SalePrice"

IDENTIFIER_COLUMNS = (
    "Order",
    "PID",
)

TRANSACTION_CONTEXT_COLUMNS = (
    "Mo Sold",
    "Yr Sold",
    "Sale Type",
    "Sale Condition",
)

PRIMARY_EXCLUDED_COLUMNS = (
    TARGET_COLUMN,
    *IDENTIFIER_COLUMNS,
    *TRANSACTION_CONTEXT_COLUMNS,
)

CATEGORICAL_CAST_COLUMNS = (
    "MS SubClass",
)


@dataclass(frozen=True)
class PrimaryModelData:
    """Prepared data partitions for the primary pre-sale model."""

    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame


def validate_primary_model_columns(data: pd.DataFrame) -> None:
    """Validate columns required to construct the primary model data."""
    required_columns = set(
        PRIMARY_EXCLUDED_COLUMNS
        + CATEGORICAL_CAST_COLUMNS
    )

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        names = ", ".join(missing_columns)

        raise ValueError(
            "Dataset is missing columns required by the primary "
            f"feature schema: {names}."
        )


def prepare_primary_model_data(
    data: pd.DataFrame,
) -> PrimaryModelData:
    """Create leakage-safe features, target, and identifier metadata.

    The input DataFrame is not modified.
    """
    validate_primary_model_columns(data)

    prepared = data.copy()

    for column in CATEGORICAL_CAST_COLUMNS:
        prepared[column] = prepared[column].astype("string")

    target = prepared[TARGET_COLUMN].copy()

    metadata = prepared.loc[
        :,
        list(IDENTIFIER_COLUMNS),
    ].copy()

    features = prepared.drop(
        columns=list(PRIMARY_EXCLUDED_COLUMNS)
    ).copy()

    prohibited_columns = set(
        PRIMARY_EXCLUDED_COLUMNS
    ).intersection(features.columns)

    if prohibited_columns:
        names = ", ".join(sorted(prohibited_columns))

        raise RuntimeError(
            "Excluded columns entered the primary feature matrix: "
            f"{names}."
        )

    if len(features) != len(target):
        raise RuntimeError(
            "Feature and target row counts do not match."
        )

    return PrimaryModelData(
        features=features,
        target=target,
        metadata=metadata,
    )
