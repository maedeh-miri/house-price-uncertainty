"""Build the reproducible evaluation-split manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from house_price_uncertainty.data import load_ames_housing
from house_price_uncertainty.splitting import (
    ROW_ID_COLUMN,
    TIME_COLUMN,
    EvaluationSplit,
    make_random_evaluation_split,
    make_temporal_evaluation_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "evaluation_split_manifest.csv"
)

PARTITION_NAMES = (
    "train",
    "calibration",
    "test",
)

MANIFEST_METADATA_COLUMNS = (
    ROW_ID_COLUMN,
    TIME_COLUMN,
)


def _build_partition_membership(
    split: EvaluationSplit,
    *,
    column_name: str,
) -> pd.DataFrame:
    """Map each stable row ID to one partition."""
    partitions = {
        "train": split.train,
        "calibration": split.calibration,
        "test": split.test,
    }

    membership_frames: list[pd.DataFrame] = []

    for partition_name in PARTITION_NAMES:
        membership_frames.append(
            partitions[partition_name][
                [ROW_ID_COLUMN]
            ].assign(
                **{
                    column_name: partition_name,
                }
            )
        )

    membership = pd.concat(
        membership_frames,
        ignore_index=True,
    )

    if not membership[ROW_ID_COLUMN].is_unique:
        raise RuntimeError(
            f"{column_name} contains duplicate row IDs."
        )

    return membership


def _validate_manifest(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    """Validate manifest completeness and partition labels."""
    expected_columns = {
        *MANIFEST_METADATA_COLUMNS,
        "random_partition",
        "temporal_partition",
    }

    if set(manifest.columns) != expected_columns:
        raise RuntimeError(
            "Evaluation manifest has unexpected columns."
        )

    if len(manifest) != len(data):
        raise RuntimeError(
            "Evaluation manifest row count does not "
            "match the dataset."
        )

    if not manifest[ROW_ID_COLUMN].is_unique:
        raise RuntimeError(
            "Evaluation manifest row IDs are not unique."
        )

    expected_ids = set(data[ROW_ID_COLUMN])
    manifest_ids = set(manifest[ROW_ID_COLUMN])

    if manifest_ids != expected_ids:
        raise RuntimeError(
            "Evaluation manifest does not cover every "
            "dataset row."
        )

    for column_name in (
        "random_partition",
        "temporal_partition",
    ):
        if manifest[column_name].isna().any():
            raise RuntimeError(
                f"{column_name} contains missing labels."
            )

        observed_labels = set(
            manifest[column_name].unique()
        )

        expected_labels = set(PARTITION_NAMES)

        if observed_labels != expected_labels:
            raise RuntimeError(
                f"{column_name} has unexpected "
                "partition labels."
            )


def build_evaluation_manifest(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create exact random and temporal row assignments."""
    missing_columns = sorted(
        set(MANIFEST_METADATA_COLUMNS).difference(
            data.columns
        )
    )

    if missing_columns:
        names = ", ".join(missing_columns)

        raise ValueError(
            "Dataset is missing manifest columns: "
            f"{names}."
        )

    random_split = make_random_evaluation_split(
        data
    )

    temporal_split = make_temporal_evaluation_split(
        data
    )

    random_membership = (
        _build_partition_membership(
            random_split,
            column_name="random_partition",
        )
    )

    temporal_membership = (
        _build_partition_membership(
            temporal_split,
            column_name="temporal_partition",
        )
    )

    manifest = (
        data.loc[
            :,
            list(MANIFEST_METADATA_COLUMNS),
        ]
        .merge(
            random_membership,
            on=ROW_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
        .merge(
            temporal_membership,
            on=ROW_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
        .sort_values(ROW_ID_COLUMN)
        .reset_index(drop=True)
    )

    _validate_manifest(
        data,
        manifest,
    )

    return manifest


def print_manifest_summary(
    manifest: pd.DataFrame,
) -> None:
    """Print partition counts and temporal year coverage."""
    partition_counts = pd.DataFrame(
        {
            "random": (
                manifest[
                    "random_partition"
                ].value_counts()
            ),
            "temporal": (
                manifest[
                    "temporal_partition"
                ].value_counts()
            ),
        }
    ).reindex(PARTITION_NAMES)

    print("=== PARTITION COUNTS ===")
    print(partition_counts.to_string())

    print("\n=== TEMPORAL YEAR COUNTS ===")

    temporal_year_counts = pd.crosstab(
        manifest["temporal_partition"],
        manifest[TIME_COLUMN],
    ).reindex(PARTITION_NAMES)

    print(temporal_year_counts.to_string())


def main() -> None:
    """Generate and save the evaluation manifest."""
    data = load_ames_housing()

    manifest = build_evaluation_manifest(data)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_PATH,
        index=False,
        lineterminator="\n",
    )

    print_manifest_summary(manifest)

    print(
        "\nSaved evaluation manifest to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Manifest rows: {len(manifest)}"
    )


if __name__ == "__main__":
    main()