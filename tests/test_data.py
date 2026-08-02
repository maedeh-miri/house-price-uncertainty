from pathlib import Path

import pandas as pd
import pytest

from house_price_uncertainty.data import load_ames_housing


def write_test_dataset(path: Path, **overrides: object) -> None:
    data: dict[str, object] = {
        "SalePrice": [100_000, 200_000],
        "Neighborhood": ["NAmes", "CollgCr"],
    }
    data.update(overrides)
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def test_loads_valid_tab_separated_data(tmp_path: Path) -> None:
    path = tmp_path / "AmesHousing.txt"
    write_test_dataset(path)

    data = load_ames_housing(
        path,
        verify_checksum=False,
        validate_source_shape=False,
    )

    assert data.shape == (2, 2)
    assert data["SalePrice"].tolist() == [100_000, 200_000]


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download instructions"):
        load_ames_housing(tmp_path / "missing.txt")


def test_rejects_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "AmesHousing.txt"
    pd.DataFrame({"SalePrice": [100_000]}).to_csv(
        path,
        sep="\t",
        index=False,
    )

    with pytest.raises(ValueError, match="Neighborhood"):
        load_ames_housing(
            path,
            verify_checksum=False,
            validate_source_shape=False,
        )


def test_rejects_non_positive_target(tmp_path: Path) -> None:
    path = tmp_path / "AmesHousing.txt"
    write_test_dataset(path, SalePrice=[100_000, 0])

    with pytest.raises(ValueError, match="positive"):
        load_ames_housing(
            path,
            verify_checksum=False,
            validate_source_shape=False,
        )


def test_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "AmesHousing.txt"
    write_test_dataset(path)

    with pytest.raises(ValueError, match="checksum"):
        load_ames_housing(path)