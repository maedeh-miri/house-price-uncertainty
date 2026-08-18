"""Integration tests for the leakage-safe preprocessing workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from house_price_uncertainty.feature_schema import prepare_primary_model_data
from house_price_uncertainty.preprocessing import build_primary_preprocessor
from house_price_uncertainty.splitting import make_random_evaluation_split


def make_raw_modeling_data() -> pd.DataFrame:
    """Create raw-like rows containing split, target, and predictor columns."""
    rows = 12

    return pd.DataFrame(
        {
            "Order": range(1, rows + 1),
            "PID": range(1001, 1001 + rows),
            "Mo Sold": [1, 2, 3, 4, 5, 6] * 2,
            "Yr Sold": [2006, 2007, 2008, 2009, 2010, 2008] * 2,
            "Sale Type": ["WD "] * rows,
            "Sale Condition": ["Normal"] * rows,
            "SalePrice": np.linspace(100000, 210000, rows),
            "MS SubClass": [20, 60, 20, 120, 50, 70] * 2,
            "Neighborhood": ["NAmes", "OldTown", "Edwards"] * 4,
            "Lot Config": ["Inside", "Corner", "CulDSac"] * 4,
            "Lot Frontage": [
                80.0,
                65.0,
                np.nan,
                75.0,
                90.0,
                70.0,
            ]
            * 2,
            "Lot Area": [
                9000.0,
                8000.0,
                10000.0,
                9500.0,
                np.nan,
                8500.0,
            ]
            * 2,
            "Overall Qual": [5, 6, 7, 5, 8, 4] * 2,
            "Garage Type": [
                "Attchd",
                "Detchd",
                "NA",
                "Attchd",
                "Detchd",
                "Attchd",
            ]
            * 2,
            "Garage Yr Blt": [
                2000.0,
                1975.0,
                np.nan,
                1990.0,
                1985.0,
                np.nan,
            ]
            * 2,
            "Mas Vnr Type": [
                "None",
                "BrkFace",
                "Stone",
                np.nan,
                "None",
                "BrkFace",
            ]
            * 2,
            "Exter Qual": ["TA", "Gd", "TA", "Fa", "Ex", "TA"] * 2,
        }
    )


def _dense_values(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()

    return np.asarray(matrix)


def test_random_split_to_preprocessing_is_train_fit_only() -> None:
    raw = make_raw_modeling_data()

    split = make_random_evaluation_split(raw)

    train = prepare_primary_model_data(split.train)
    calibration = prepare_primary_model_data(split.calibration)
    test = prepare_primary_model_data(split.test)

    preprocessor = build_primary_preprocessor(scale_numeric=True)

    train_matrix = preprocessor.fit_transform(train.features)

    numeric = (
        preprocessor.named_steps["columns"]
        .named_transformers_["numeric"]
    )

    scaler_mean_before = (
        numeric.named_steps["scaler"]
        .mean_
        .copy()
    )

    calibration_matrix = preprocessor.transform(
        calibration.features
    )

    test_matrix = preprocessor.transform(
        test.features
    )

    scaler_mean_after = (
        numeric.named_steps["scaler"]
        .mean_
        .copy()
    )

    assert train_matrix.shape[0] == len(split.train)
    assert calibration_matrix.shape[0] == len(split.calibration)
    assert test_matrix.shape[0] == len(split.test)

    assert (
        train_matrix.shape[1]
        == calibration_matrix.shape[1]
        == test_matrix.shape[1]
    )

    assert np.array_equal(
        scaler_mean_before,
        scaler_mean_after,
    )

    assert np.isfinite(
        _dense_values(train_matrix)
    ).all()

    assert np.isfinite(
        _dense_values(calibration_matrix)
    ).all()

    assert np.isfinite(
        _dense_values(test_matrix)
    ).all()