"""Tests for leakage-safe preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from house_price_uncertainty.preprocessing import (
    GarageYearImputer,
    HierarchicalLotFrontageImputer,
    build_primary_preprocessor,
    validate_preprocessing_features,
)


def make_train_features() -> pd.DataFrame:
    """Create a small prepared feature matrix with mixed feature types."""
    return pd.DataFrame(
        {
            "MS SubClass": pd.Series(["20", "60", "20", "120"], dtype="string"),
            "Neighborhood": ["NAmes", "NAmes", "OldTown", "OldTown"],
            "Lot Config": ["Inside", "Corner", "Inside", "Corner"],
            "Lot Frontage": [80.0, 100.0, 60.0, np.nan],
            "Lot Area": [9000.0, 11000.0, 7500.0, np.nan],
            "Overall Qual": [5, 7, 4, 6],
            "Garage Type": ["Attchd", "NA", "Detchd", "Attchd"],
            "Garage Yr Blt": [2000.0, np.nan, 1970.0, np.nan],
            "Mas Vnr Type": ["None", "BrkFace", np.nan, "Stone"],
            "Exter Qual": ["TA", "Gd", "Fa", "Ex"],
        }
    )


def make_evaluation_features() -> pd.DataFrame:
    """Create evaluation rows containing missing and unseen categories."""
    return pd.DataFrame(
        {
            "MS SubClass": pd.Series(["150", "20"], dtype="string"),
            "Neighborhood": ["NAmes", "NewNeighborhood"],
            "Lot Config": ["Inside", "CulDSac"],
            "Lot Frontage": [np.nan, np.nan],
            "Lot Area": [np.nan, 15000.0],
            "Overall Qual": [8, 3],
            "Garage Type": ["Attchd", "NA"],
            "Garage Yr Blt": [np.nan, np.nan],
            "Mas Vnr Type": [np.nan, "NewType"],
            "Exter Qual": ["Gd", "Po"],
        }
    )


def test_hierarchical_lot_frontage_uses_train_only_statistics() -> None:
    train = make_train_features()
    evaluation = make_evaluation_features()

    imputer = HierarchicalLotFrontageImputer().fit(train)
    transformed = imputer.transform(evaluation)

    # NAmes training median is (80 + 100) / 2 = 90.
    assert transformed.loc[0, "Lot Frontage"] == pytest.approx(90.0)
    # Unseen neighborhood + unseen lot config falls back to train global median = 80.
    assert transformed.loc[1, "Lot Frontage"] == pytest.approx(80.0)


def test_hierarchical_lot_frontage_does_not_modify_input() -> None:
    train = make_train_features()
    original = train.copy(deep=True)

    HierarchicalLotFrontageImputer().fit_transform(train)

    pd.testing.assert_frame_equal(train, original)


def test_garage_year_distinguishes_absence_from_unknown_missing() -> None:
    train = make_train_features()
    evaluation = make_evaluation_features()

    imputer = GarageYearImputer().fit(train)
    transformed = imputer.transform(evaluation)

    # Unknown year for an existing garage uses the train-only median: (2000 + 1970) / 2.
    assert transformed.loc[0, "Garage Yr Blt"] == pytest.approx(1985.0)
    # Structural absence is represented distinctly with zero.
    assert transformed.loc[1, "Garage Yr Blt"] == pytest.approx(0.0)


def test_primary_preprocessor_handles_unseen_categories_and_missing_values() -> None:
    train = make_train_features()
    evaluation = make_evaluation_features()

    preprocessor = build_primary_preprocessor(scale_numeric=True)
    train_matrix = preprocessor.fit_transform(train)
    evaluation_matrix = preprocessor.transform(evaluation)

    assert train_matrix.shape[0] == len(train)
    assert evaluation_matrix.shape[0] == len(evaluation)
    assert train_matrix.shape[1] == evaluation_matrix.shape[1]
    train_values = (
        train_matrix.toarray()
        if sparse.issparse(train_matrix)
        else np.asarray(train_matrix)
    )
    evaluation_values = (
        evaluation_matrix.toarray()
        if sparse.issparse(evaluation_matrix)
        else np.asarray(evaluation_matrix)
    )
    assert np.isfinite(train_values).all()
    assert np.isfinite(evaluation_values).all()


def test_structural_na_is_preserved_as_a_categorical_level() -> None:
    train = make_train_features()

    preprocessor = build_primary_preprocessor(scale_numeric=False)
    preprocessor.fit(train)

    feature_names = preprocessor.named_steps["columns"].get_feature_names_out()

    assert "Garage Type_NA" in feature_names


def test_numeric_scaling_is_fitted_from_training_data_only() -> None:
    train = make_train_features()

    preprocessor = build_primary_preprocessor(scale_numeric=True)
    preprocessor.fit(train)

    numeric = preprocessor.named_steps["columns"].named_transformers_["numeric"]
    scaler = numeric.named_steps["scaler"]
    numeric_columns = preprocessor.named_steps["columns"].transformers_[0][2]

    lot_area_position = list(numeric_columns).index("Lot Area")
    # Train-only median imputes Lot Area as 9000 before scaling.
    expected_mean = np.mean([9000.0, 11000.0, 7500.0, 9000.0])
    assert scaler.mean_[lot_area_position] == pytest.approx(expected_mean)


def test_preprocessor_rejects_excluded_columns() -> None:
    features = make_train_features().assign(SalePrice=[1, 2, 3, 4])

    with pytest.raises(ValueError, match="excluded columns"):
        validate_preprocessing_features(features)


def test_preprocessor_rejects_column_drift_after_fit() -> None:
    train = make_train_features()
    evaluation = make_evaluation_features().drop(columns="Exter Qual")

    preprocessor = build_primary_preprocessor()
    preprocessor.fit(train)

    with pytest.raises(ValueError, match="feature columns changed"):
        preprocessor.transform(evaluation)
