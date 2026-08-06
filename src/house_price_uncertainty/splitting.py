"""Deterministic, leakage-safe evaluation splits for Ames Housing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

ROW_ID_COLUMN = "Order"
TIME_COLUMN = "Yr Sold"

FIRST_RANDOM_STATE = 42
SECOND_RANDOM_STATE = 43

RANDOM_REMAINDER_FRACTION = 0.40
REMAINDER_TEST_FRACTION = 0.50

TEMPORAL_TRAIN_END_YEAR = 2008
TEMPORAL_CALIBRATION_YEAR = 2009
TEMPORAL_TEST_YEAR = 2010


@dataclass(frozen=True)
class EvaluationSplit:
    """Train, calibration, and test partitions."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def validate_source_rows(
    data: pd.DataFrame,
    *,
    require_time_column: bool = False,
) -> None:
    """Validate columns and values required for partitioning."""
    required_columns = {ROW_ID_COLUMN}

    if require_time_column:
        required_columns.add(TIME_COLUMN)

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        names = ", ".join(missing_columns)

        raise ValueError(
            f"Dataset is missing split columns: {names}."
        )

    if data.empty:
        raise ValueError(
            "Dataset must contain at least one row."
        )

    if data[ROW_ID_COLUMN].isna().any():
        raise ValueError(
            f"{ROW_ID_COLUMN} must not contain missing values."
        )

    if not data[ROW_ID_COLUMN].is_unique:
        raise ValueError(
            f"{ROW_ID_COLUMN} must uniquely identify every row."
        )

    if require_time_column:
        if data[TIME_COLUMN].isna().any():
            raise ValueError(
                f"{TIME_COLUMN} must not contain missing values."
            )

        if not pd.api.types.is_numeric_dtype(
            data[TIME_COLUMN]
        ):
            raise ValueError(
                f"{TIME_COLUMN} must be numeric."
            )


def validate_evaluation_split(
    source: pd.DataFrame,
    split: EvaluationSplit,
) -> None:
    """Validate that partitions are disjoint and exhaustive."""
    validate_source_rows(source)

    partitions = {
        "train": split.train,
        "calibration": split.calibration,
        "test": split.test,
    }

    partition_ids: dict[str, set[object]] = {}

    for name, partition in partitions.items():
        validate_source_rows(partition)

        partition_ids[name] = set(
            partition[ROW_ID_COLUMN]
        )

    partition_pairs = (
        ("train", "calibration"),
        ("train", "test"),
        ("calibration", "test"),
    )

    for left_name, right_name in partition_pairs:
        overlap = (
            partition_ids[left_name]
            & partition_ids[right_name]
        )

        if overlap:
            raise RuntimeError(
                f"{left_name} and {right_name} overlap by "
                f"{len(overlap)} rows."
            )

    source_ids = set(source[ROW_ID_COLUMN])

    combined_ids = set().union(
        *partition_ids.values()
    )

    if combined_ids != source_ids:
        missing_ids = source_ids.difference(
            combined_ids
        )

        unexpected_ids = combined_ids.difference(
            source_ids
        )

        raise RuntimeError(
            "Evaluation partitions do not match the "
            "source rows. "
            f"Missing IDs: {len(missing_ids)}. "
            f"Unexpected IDs: {len(unexpected_ids)}."
        )

    total_partition_rows = sum(
        len(partition)
        for partition in partitions.values()
    )

    if total_partition_rows != len(source):
        raise RuntimeError(
            "Evaluation partition row counts do not sum "
            "to the source row count."
        )


def _select_rows_by_id(
    data: pd.DataFrame,
    row_ids: pd.Series,
) -> pd.DataFrame:
    """Select rows by stable IDs with deterministic ordering."""
    return (
        data.loc[
            data[ROW_ID_COLUMN].isin(row_ids)
        ]
        .sort_values(ROW_ID_COLUMN)
        .reset_index(drop=True)
    )


def make_random_evaluation_split(
    data: pd.DataFrame,
    *,
    first_random_state: int = FIRST_RANDOM_STATE,
    second_random_state: int = SECOND_RANDOM_STATE,
) -> EvaluationSplit:
    """Create the primary target-independent 60/20/20 split."""
    validate_source_rows(data)

    if len(data) < 5:
        raise ValueError(
            "Random evaluation splitting requires "
            "at least five rows."
        )

    ordered_row_ids = (
        data[ROW_ID_COLUMN]
        .sort_values()
        .reset_index(drop=True)
    )

    train_ids, remainder_ids = train_test_split(
        ordered_row_ids,
        test_size=RANDOM_REMAINDER_FRACTION,
        random_state=first_random_state,
        shuffle=True,
    )

    calibration_ids, test_ids = train_test_split(
        remainder_ids,
        test_size=REMAINDER_TEST_FRACTION,
        random_state=second_random_state,
        shuffle=True,
    )

    split = EvaluationSplit(
        train=_select_rows_by_id(
            data,
            train_ids,
        ),
        calibration=_select_rows_by_id(
            data,
            calibration_ids,
        ),
        test=_select_rows_by_id(
            data,
            test_ids,
        ),
    )

    validate_evaluation_split(
        data,
        split,
    )

    return split


def make_temporal_evaluation_split(
    data: pd.DataFrame,
    *,
    train_end_year: int = TEMPORAL_TRAIN_END_YEAR,
    calibration_year: int = (
        TEMPORAL_CALIBRATION_YEAR
    ),
    test_year: int = TEMPORAL_TEST_YEAR,
) -> EvaluationSplit:
    """Create the forward-looking temporal stress-test split."""
    validate_source_rows(
        data,
        require_time_column=True,
    )

    if not (
        train_end_year
        < calibration_year
        < test_year
    ):
        raise ValueError(
            "Temporal years must satisfy "
            "train_end_year < calibration_year "
            "< test_year."
        )

    covered_rows = (
        data[TIME_COLUMN].le(train_end_year)
        | data[TIME_COLUMN].eq(calibration_year)
        | data[TIME_COLUMN].eq(test_year)
    )

    if not covered_rows.all():
        uncovered_years = sorted(
            data.loc[
                ~covered_rows,
                TIME_COLUMN,
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Temporal split leaves rows unassigned. "
            f"Uncovered years: {uncovered_years}."
        )

    split = EvaluationSplit(
        train=(
            data.loc[
                data[TIME_COLUMN].le(
                    train_end_year
                )
            ]
            .sort_values(ROW_ID_COLUMN)
            .reset_index(drop=True)
        ),
        calibration=(
            data.loc[
                data[TIME_COLUMN].eq(
                    calibration_year
                )
            ]
            .sort_values(ROW_ID_COLUMN)
            .reset_index(drop=True)
        ),
        test=(
            data.loc[
                data[TIME_COLUMN].eq(
                    test_year
                )
            ]
            .sort_values(ROW_ID_COLUMN)
            .reset_index(drop=True)
        ),
    )

    empty_partitions = [
        name
        for name, partition in {
            "train": split.train,
            "calibration": split.calibration,
            "test": split.test,
        }.items()
        if partition.empty
    ]

    if empty_partitions:
        names = ", ".join(empty_partitions)

        raise ValueError(
            "Temporal split produced empty partitions: "
            f"{names}."
        )

    validate_evaluation_split(
        data,
        split,
    )

    return split