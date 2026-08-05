"""Tests for leakage-safe primary feature preparation."""

from __future__ import annotations

import pandas as pd
import pytest

from house_price_uncertainty.feature_schema import (
    IDENTIFIER_COLUMNS,
    PRIMARY_EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    TRANSACTION_CONTEXT_COLUMNS,
    prepare_primary_model_data,
)


def make_sample_data() -> pd.DataFrame:
    """Create a small dataset containing all required feature roles."""
    return pd.DataFrame(
        {
            "Order": [1, 2, 3],
            "PID": [1001, 1002, 1003],
            "MS SubClass": [20, 60, 120],
            "Neighborhood": ["NAmes", "CollgCr", "OldTown"],
            "Lot Area": [9000, 11000, 7500],
            "Overall Qual": [5, 7, 4],
            "Mo Sold": [5, 7, 10],
            "Yr Sold": [2008, 2009, 2010],
            "Sale Type": ["WD", "New", "WD"],
            "Sale Condition": [
                "Normal",
                "Partial",
                "Abnorml",
            ],
            "SalePrice": [150000, 250000, 110000],
        }
    )


def test_prepare_primary_model_data_excludes_prohibited_columns() -> None:
    """Excluded columns must never enter the primary feature matrix."""
    data = make_sample_data()

    prepared = prepare_primary_model_data(data)

    prohibited = set(PRIMARY_EXCLUDED_COLUMNS)

    assert prohibited.isdisjoint(prepared.features.columns)


def test_prepare_primary_model_data_preserves_allowed_features() -> None:
    """Property predictors should remain in the feature matrix."""
    data = make_sample_data()

    prepared = prepare_primary_model_data(data)

    assert list(prepared.features.columns) == [
        "MS SubClass",
        "Neighborhood",
        "Lot Area",
        "Overall Qual",
    ]


def test_prepare_primary_model_data_returns_target() -> None:
    """The target output must equal the original SalePrice column."""
    data = make_sample_data()

    prepared = prepare_primary_model_data(data)

    pd.testing.assert_series_equal(
        prepared.target,
        data[TARGET_COLUMN],
    )


def test_prepare_primary_model_data_returns_identifier_metadata() -> None:
    """Identifiers should remain available outside the feature matrix."""
    data = make_sample_data()

    prepared = prepare_primary_model_data(data)

    pd.testing.assert_frame_equal(
        prepared.metadata,
        data.loc[:, list(IDENTIFIER_COLUMNS)],
    )


def test_ms_subclass_is_cast_to_string() -> None:
    """MS SubClass codes must not remain ordinary numeric measurements."""
    data = make_sample_data()

    prepared = prepare_primary_model_data(data)

    assert isinstance(
        prepared.features["MS SubClass"].dtype,
        pd.StringDtype,
    )

    assert prepared.features["MS SubClass"].tolist() == [
        "20",
        "60",
        "120",
    ]


def test_prepare_primary_model_data_does_not_modify_input() -> None:
    """Feature preparation must not mutate the source DataFrame."""
    data = make_sample_data()
    original = data.copy(deep=True)

    prepare_primary_model_data(data)

    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "missing_column",
    [
        TARGET_COLUMN,
        *IDENTIFIER_COLUMNS,
        *TRANSACTION_CONTEXT_COLUMNS,
        "MS SubClass",
    ],
)
def test_missing_required_column_raises(
    missing_column: str,
) -> None:
    """Missing schema columns should cause an explicit failure."""
    data = make_sample_data().drop(columns=missing_column)

    with pytest.raises(
        ValueError,
        match="missing columns required",
    ):
        prepare_primary_model_data(data)
