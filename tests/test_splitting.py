"""Tests for deterministic evaluation splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from house_price_uncertainty.splitting import (
    ROW_ID_COLUMN,
    TIME_COLUMN,
    EvaluationSplit,
    make_random_evaluation_split,
    make_temporal_evaluation_split,
)


def make_sample_data() -> pd.DataFrame:
    """Create 50 rows covering five sale years."""
    rows = 50

    return pd.DataFrame(
        {
            ROW_ID_COLUMN: range(
                1,
                rows + 1,
            ),
            "PID": range(
                1001,
                1001 + rows,
            ),
            TIME_COLUMN: (
                [2006] * 10
                + [2007] * 10
                + [2008] * 10
                + [2009] * 10
                + [2010] * 10
            ),
            "Neighborhood": [
                "NAmes",
                "CollgCr",
                "OldTown",
                "Sawyer",
                "Edwards",
            ]
            * 10,
            "SalePrice": [
                100000 + index * 2500
                for index in range(rows)
            ],
        }
    )


def partition_ids(
    split: EvaluationSplit,
) -> tuple[
    set[int],
    set[int],
    set[int],
]:
    """Return stable row-ID sets from a split."""
    return (
        set(split.train[ROW_ID_COLUMN]),
        set(split.calibration[ROW_ID_COLUMN]),
        set(split.test[ROW_ID_COLUMN]),
    )


def test_random_split_has_expected_sizes() -> None:
    """The primary protocol should create 60/20/20."""
    split = make_random_evaluation_split(
        make_sample_data()
    )

    assert len(split.train) == 30
    assert len(split.calibration) == 10
    assert len(split.test) == 10


def test_random_split_is_disjoint_and_exhaustive() -> None:
    """Every row should appear in exactly one partition."""
    data = make_sample_data()

    split = make_random_evaluation_split(data)

    train_ids, calibration_ids, test_ids = (
        partition_ids(split)
    )

    assert train_ids.isdisjoint(
        calibration_ids
    )
    assert train_ids.isdisjoint(
        test_ids
    )
    assert calibration_ids.isdisjoint(
        test_ids
    )

    combined_ids = (
        train_ids
        | calibration_ids
        | test_ids
    )

    assert combined_ids == set(
        data[ROW_ID_COLUMN]
    )


def test_random_split_is_deterministic() -> None:
    """Fixed seeds should reproduce identical memberships."""
    data = make_sample_data()

    first = make_random_evaluation_split(data)
    second = make_random_evaluation_split(data)

    assert (
        partition_ids(first)
        == partition_ids(second)
    )


def test_random_split_is_independent_of_source_order() -> None:
    """Reordering rows must not change split membership."""
    data = make_sample_data()

    shuffled = (
        data.sample(
            frac=1,
            random_state=999,
        )
        .reset_index(drop=True)
    )

    original = make_random_evaluation_split(data)

    reordered = make_random_evaluation_split(
        shuffled
    )

    assert (
        partition_ids(original)
        == partition_ids(reordered)
    )


def test_random_split_does_not_require_target() -> None:
    """The primary split must not depend on SalePrice."""
    data = make_sample_data()

    without_target = data.drop(
        columns="SalePrice"
    )

    full = make_random_evaluation_split(data)

    target_free = make_random_evaluation_split(
        without_target
    )

    assert (
        partition_ids(full)
        == partition_ids(target_free)
    )


def test_partitions_have_stable_order_and_index() -> None:
    """Partitions should be ordered by ID with reset indices."""
    split = make_random_evaluation_split(
        make_sample_data()
    )

    for partition in (
        split.train,
        split.calibration,
        split.test,
    ):
        assert partition[
            ROW_ID_COLUMN
        ].is_monotonic_increasing

        assert partition.index.tolist() == list(
            range(len(partition))
        )


def test_temporal_split_uses_forward_years() -> None:
    """Temporal partitions must respect chronological order."""
    data = make_sample_data()

    split = make_temporal_evaluation_split(data)

    assert (
        split.train[TIME_COLUMN].max()
        == 2008
    )

    assert set(
        split.calibration[TIME_COLUMN]
    ) == {2009}

    assert set(
        split.test[TIME_COLUMN]
    ) == {2010}

    assert len(split.train) == 30
    assert len(split.calibration) == 10
    assert len(split.test) == 10

    combined_ids = set().union(
        *partition_ids(split)
    )

    assert combined_ids == set(
        data[ROW_ID_COLUMN]
    )


def test_missing_row_identifier_raises() -> None:
    """A stable row identifier is required."""
    data = make_sample_data().drop(
        columns=ROW_ID_COLUMN
    )

    with pytest.raises(
        ValueError,
        match="missing split columns",
    ):
        make_random_evaluation_split(data)


def test_duplicate_row_identifier_raises() -> None:
    """Duplicate row identifiers must fail."""
    data = make_sample_data()

    data.loc[
        1,
        ROW_ID_COLUMN,
    ] = data.loc[
        0,
        ROW_ID_COLUMN,
    ]

    with pytest.raises(
        ValueError,
        match="uniquely identify",
    ):
        make_random_evaluation_split(data)


def test_temporal_split_requires_time_column() -> None:
    """The temporal protocol requires Yr Sold."""
    data = make_sample_data().drop(
        columns=TIME_COLUMN
    )

    with pytest.raises(
        ValueError,
        match="missing split columns",
    ):
        make_temporal_evaluation_split(data)


def test_temporal_split_rejects_missing_years() -> None:
    """Missing sale years must fail before partitioning."""
    data = make_sample_data()

    data[TIME_COLUMN] = data[
        TIME_COLUMN
    ].astype("Float64")

    data.loc[
        0,
        TIME_COLUMN,
    ] = pd.NA

    with pytest.raises(
        ValueError,
        match="must not contain missing",
    ):
        make_temporal_evaluation_split(data)


def test_temporal_split_rejects_nonnumeric_years() -> None:
    """Sale years must use a numeric dtype."""
    data = make_sample_data()

    data[TIME_COLUMN] = data[
        TIME_COLUMN
    ].astype(str)

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        make_temporal_evaluation_split(data)


def test_temporal_split_rejects_uncovered_years() -> None:
    """Every source year must belong to a partition."""
    data = make_sample_data()

    data.loc[
        0,
        TIME_COLUMN,
    ] = 2011

    with pytest.raises(
        ValueError,
        match="leaves rows unassigned",
    ):
        make_temporal_evaluation_split(data)


def test_temporal_split_rejects_invalid_year_order() -> None:
    """Temporal boundaries must be strictly increasing."""
    data = make_sample_data()

    with pytest.raises(
        ValueError,
        match="must satisfy",
    ):
        make_temporal_evaluation_split(
            data,
            train_end_year=2009,
            calibration_year=2008,
            test_year=2010,
        )