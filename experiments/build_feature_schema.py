"""Build the reviewed feature schema for the Ames Housing dataset."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from house_price_uncertainty.data import load_ames_housing

OUTPUT_PATH = Path("reports/feature_schema.csv")
EXPECTED_COLUMN_COUNT = 82


# Columns used only for tracing and record identification.
METADATA_COLUMNS = frozenset(
    {
        "Order",
        "PID",
    }
)


# Prediction target.
TARGET_COLUMNS = frozenset(
    {
        "SalePrice",
    }
)


# Variables describing the realized sale transaction.
#
# These are excluded from the primary pre-sale model because their
# availability is not guaranteed at valuation time.
TRANSACTION_COLUMNS = frozenset(
    {
        "Mo Sold",
        "Yr Sold",
        "Sale Type",
        "Sale Condition",
    }
)


# Property characteristics that usually require an inspection,
# assessment, or structured quality evaluation.
INSPECTION_COLUMNS = frozenset(
    {
        "Overall Qual",
        "Overall Cond",
        "Exter Qual",
        "Exter Cond",
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",
        "Heating QC",
        "Kitchen Qual",
        "Functional",
        "Fireplace Qu",
        "Garage Finish",
        "Garage Qual",
        "Garage Cond",
        "Pool QC",
    }
)


# Nominal categories have labels but no defensible numeric order.
NOMINAL_COLUMNS = frozenset(
    {
        "MS SubClass",
        "MS Zoning",
        "Street",
        "Alley",
        "Land Contour",
        "Lot Config",
        "Neighborhood",
        "Condition 1",
        "Condition 2",
        "Bldg Type",
        "House Style",
        "Roof Style",
        "Roof Matl",
        "Exterior 1st",
        "Exterior 2nd",
        "Mas Vnr Type",
        "Foundation",
        "Heating",
        "Central Air",
        "Electrical",
        "Garage Type",
        "Fence",
        "Misc Feature",
        "Sale Type",
        "Sale Condition",
    }
)


# Ordinal variables have categories or scores with a meaningful order.
ORDINAL_COLUMNS = frozenset(
    {
        "Lot Shape",
        "Utilities",
        "Land Slope",
        "Overall Qual",
        "Overall Cond",
        "Exter Qual",
        "Exter Cond",
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",
        "Heating QC",
        "Kitchen Qual",
        "Functional",
        "Fireplace Qu",
        "Garage Finish",
        "Garage Qual",
        "Garage Cond",
        "Paved Drive",
        "Pool QC",
    }
)


# Month is cyclical because December and January are adjacent in time.
TEMPORAL_CYCLICAL_COLUMNS = frozenset(
    {
        "Mo Sold",
    }
)


# Year has a chronological order but is not an ordinary property
# measurement.
TEMPORAL_ORDERED_COLUMNS = frozenset(
    {
        "Yr Sold",
    }
)


def find_overlaps(
    groups: Mapping[str, frozenset[str]],
) -> dict[str, list[str]]:
    """Return columns assigned to more than one group."""
    assignments: dict[str, list[str]] = {}

    for group_name, columns in groups.items():
        for column in columns:
            assignments.setdefault(column, []).append(group_name)

    return {
        column: group_names
        for column, group_names in assignments.items()
        if len(group_names) > 1
    }


def validate_configuration(data: pd.DataFrame) -> None:
    """Validate the manually reviewed schema configuration."""
    if data.shape[1] != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            "Dataset column count does not match the reviewed schema: "
            f"expected {EXPECTED_COLUMN_COUNT}, received {data.shape[1]}."
        )

    availability_groups = {
        "metadata": METADATA_COLUMNS,
        "target": TARGET_COLUMNS,
        "transaction": TRANSACTION_COLUMNS,
        "inspection": INSPECTION_COLUMNS,
    }

    availability_overlaps = find_overlaps(availability_groups)

    if availability_overlaps:
        raise ValueError(
            "Columns appear in multiple availability groups: "
            f"{availability_overlaps}."
        )

    semantic_groups = {
        "identifier": METADATA_COLUMNS,
        "target": TARGET_COLUMNS,
        "nominal categorical": NOMINAL_COLUMNS,
        "ordinal": ORDINAL_COLUMNS,
        "temporal cyclical": TEMPORAL_CYCLICAL_COLUMNS,
        "temporal ordered": TEMPORAL_ORDERED_COLUMNS,
    }

    semantic_overlaps = find_overlaps(semantic_groups)

    if semantic_overlaps:
        raise ValueError(
            "Columns appear in multiple semantic groups: "
            f"{semantic_overlaps}."
        )

    configured_columns = set().union(
        *availability_groups.values(),
        *semantic_groups.values(),
    )

    unknown_columns = sorted(
        configured_columns.difference(data.columns)
    )

    if unknown_columns:
        names = ", ".join(unknown_columns)
        raise ValueError(
            f"Configured columns are missing from the dataset: {names}."
        )


def determine_availability(
    column: str,
) -> tuple[str, str]:
    """Return the availability tier and primary-model decision."""
    if column in TARGET_COLUMNS:
        return "target", "exclude from predictors"

    if column in METADATA_COLUMNS:
        return "metadata", "exclude from predictors"

    if column in TRANSACTION_COLUMNS:
        return (
            "transaction context",
            "exclude from primary model",
        )

    if column in INSPECTION_COLUMNS:
        return (
            "inspection-dependent property",
            "keep provisionally",
        )

    return "core property", "keep provisionally"


def determine_semantic_type(
    column: str,
    series: pd.Series,
) -> str:
    """Return the reviewed semantic type of a feature."""
    if column in TARGET_COLUMNS:
        return "target"

    if column in METADATA_COLUMNS:
        return "identifier"

    if column in TEMPORAL_CYCLICAL_COLUMNS:
        return "temporal cyclical"

    if column in TEMPORAL_ORDERED_COLUMNS:
        return "temporal ordered"

    if column in NOMINAL_COLUMNS:
        return "nominal categorical"

    if column in ORDINAL_COLUMNS:
        return "ordinal"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "nominal categorical"


def build_feature_schema(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Build one reviewed schema row for every dataset column."""
    records: list[dict[str, object]] = []

    for column in data.columns:
        series = data[column]

        availability_tier, primary_model_decision = (
            determine_availability(column)
        )

        records.append(
            {
                "column": column,
                "source_dtype": str(series.dtype),
                "semantic_type": determine_semantic_type(
                    column,
                    series,
                ),
                "availability_tier": availability_tier,
                "primary_model_decision": primary_model_decision,
                "missing_count": int(series.isna().sum()),
                "unique_count": int(
                    series.nunique(dropna=False)
                ),
            }
        )

    schema = pd.DataFrame(records)

    if len(schema) != data.shape[1]:
        raise RuntimeError(
            "Feature schema does not cover every dataset column."
        )

    if schema["column"].duplicated().any():
        duplicated_columns = (
            schema.loc[
                schema["column"].duplicated(),
                "column",
            ]
            .sort_values()
            .tolist()
        )

        raise RuntimeError(
            "Feature schema contains duplicate entries: "
            f"{duplicated_columns}."
        )

    source_columns = set(data.columns)
    schema_columns = set(schema["column"])

    if schema_columns != source_columns:
        missing_from_schema = sorted(
            source_columns.difference(schema_columns)
        )
        unexpected_in_schema = sorted(
            schema_columns.difference(source_columns)
        )

        raise RuntimeError(
            "Feature schema and source dataset columns differ. "
            f"Missing from schema: {missing_from_schema}. "
            f"Unexpected in schema: {unexpected_in_schema}."
        )

    return schema


def print_schema_summary(schema: pd.DataFrame) -> None:
    """Print the main validation summaries."""
    print("=== FEATURE SCHEMA SUMMARY ===")

    print(
        schema.groupby(
            [
                "availability_tier",
                "primary_model_decision",
            ],
            observed=True,
        )
        .size()
        .rename("columns")
        .to_string()
    )

    print("\n=== SEMANTIC TYPE SUMMARY ===")

    print(
        schema["semantic_type"]
        .value_counts()
        .to_string()
    )

    print("\n=== EXCLUDED OR SPECIAL COLUMNS ===")

    special_columns = schema.loc[
        schema["primary_model_decision"].ne(
            "keep provisionally"
        ),
        [
            "column",
            "semantic_type",
            "availability_tier",
            "primary_model_decision",
        ],
    ]

    print(special_columns.to_string(index=False))


def main() -> None:
    """Build, validate, display, and save the feature schema."""
    data = load_ames_housing()

    validate_configuration(data)

    schema = build_feature_schema(data)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    schema.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_schema_summary(schema)

    print(f"\nTotal schema columns: {len(schema)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
