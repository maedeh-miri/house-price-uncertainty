"""Loading and validation utilities for the Ames Housing dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("data/raw/AmesHousing.txt")
EXPECTED_SHA256 = "6CFE6CB525BA437DE428653A1040E2AED7D696640BF75203786A6D7A0E67CFCC"
EXPECTED_ROWS = 2930
EXPECTED_COLUMNS = 82
REQUIRED_COLUMNS = frozenset({"Neighborhood", "SalePrice"})


def calculate_sha256(path: str | Path) -> str:
    """Calculate the uppercase SHA256 checksum of a file."""
    file_path = Path(path)
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest().upper()


def validate_ames_housing(data: pd.DataFrame) -> None:
    """Validate the core schema and target assumptions of the Ames data."""
    if data.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {names}.")

    if data.duplicated().any():
        raise ValueError("Dataset contains duplicate rows.")

    target = data["SalePrice"]

    if target.isna().any():
        raise ValueError("SalePrice must not contain missing values.")

    if not pd.api.types.is_numeric_dtype(target):
        raise ValueError("SalePrice must be numeric.")

    if (target <= 0).any():
        raise ValueError("SalePrice must contain only positive values.")

    neighborhoods = data["Neighborhood"]

    if neighborhoods.isna().any() or neighborhoods.astype(str).str.strip().eq("").any():
        raise ValueError("Neighborhood must not contain missing or blank values.")


def load_ames_housing(
    path: str | Path = DEFAULT_DATA_PATH,
    *,
    verify_checksum: bool = True,
    validate_source_shape: bool = True,
) -> pd.DataFrame:
    """Load the Ames Housing file and validate its expected structure."""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Ames Housing data was not found at {file_path}. "
            "See data/README.md for download instructions."
        )

    if verify_checksum:
        actual_checksum = calculate_sha256(file_path)

        if actual_checksum != EXPECTED_SHA256:
            raise ValueError(
                "Dataset checksum does not match the documented source version: "
                f"expected {EXPECTED_SHA256}, received {actual_checksum}."
            )

    data = pd.read_csv(
        file_path,
        sep="\t",
        keep_default_na=False,
        na_values=[""],
    )

    validate_ames_housing(data)

    expected_shape = (EXPECTED_ROWS, EXPECTED_COLUMNS)

    if validate_source_shape and data.shape != expected_shape:
        raise ValueError(
            "Dataset shape does not match the documented source version: "
            f"expected {expected_shape}, received {data.shape}."
        )

    return data