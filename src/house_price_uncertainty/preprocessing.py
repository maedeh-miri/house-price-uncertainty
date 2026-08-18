"""Leakage-safe preprocessing for the primary Ames Housing model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_price_uncertainty.feature_schema import PRIMARY_EXCLUDED_COLUMNS

LOT_FRONTAGE_COLUMN = "Lot Frontage"
NEIGHBORHOOD_COLUMN = "Neighborhood"
LOT_CONFIG_COLUMN = "Lot Config"
GARAGE_YEAR_COLUMN = "Garage Yr Blt"
GARAGE_TYPE_COLUMN = "Garage Type"
STRUCTURAL_ABSENCE_TOKEN = "NA"
CATEGORICAL_MISSING_TOKEN = "__MISSING__"

PREPROCESSING_REQUIRED_COLUMNS = (
    LOT_FRONTAGE_COLUMN,
    NEIGHBORHOOD_COLUMN,
    LOT_CONFIG_COLUMN,
    GARAGE_YEAR_COLUMN,
    GARAGE_TYPE_COLUMN,
)


def _require_dataframe(data: object) -> pd.DataFrame:
    """Return a DataFrame or fail with an explicit preprocessing error."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Preprocessing requires a pandas DataFrame input.")

    return data


def _validate_columns(data: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    missing_columns = sorted(set(required).difference(data.columns))

    if missing_columns:
        names = ", ".join(missing_columns)
        raise ValueError(f"{context} is missing required columns: {names}.")


def validate_preprocessing_features(data: pd.DataFrame) -> None:
    """Validate that only prepared primary-model features enter preprocessing."""
    frame = _require_dataframe(data)

    prohibited = sorted(set(PRIMARY_EXCLUDED_COLUMNS).intersection(frame.columns))
    if prohibited:
        names = ", ".join(prohibited)
        raise ValueError(
            "Primary preprocessing received excluded columns: "
            f"{names}. Run prepare_primary_model_data first."
        )

    _validate_columns(
        frame,
        PREPROCESSING_REQUIRED_COLUMNS,
        context="Primary preprocessing input",
    )


class PrimaryFeatureValidator(BaseEstimator, TransformerMixin):
    """Sklearn-compatible guard for the primary feature matrix."""

    def fit(self, X: pd.DataFrame, y: object = None) -> PrimaryFeatureValidator:
        del y
        validate_preprocessing_features(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _require_dataframe(X)
        validate_preprocessing_features(frame)

        expected = list(self.feature_names_in_)
        received = list(frame.columns)
        if received != expected:
            missing = sorted(set(expected).difference(received))
            unexpected = sorted(set(received).difference(expected))
            raise ValueError(
                "Preprocessing feature columns changed after fitting. "
                f"Missing: {missing}. Unexpected: {unexpected}."
            )

        return frame.copy()


class HierarchicalLotFrontageImputer(BaseEstimator, TransformerMixin):
    """Impute Lot Frontage using train-fitted hierarchical medians.

    Missing frontage is filled from the training partition in this order:
    Neighborhood median -> Lot Config median -> global median.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> HierarchicalLotFrontageImputer:
        del y
        frame = _require_dataframe(X)
        _validate_columns(
            frame,
            (LOT_FRONTAGE_COLUMN, NEIGHBORHOOD_COLUMN, LOT_CONFIG_COLUMN),
            context="Lot Frontage imputer input",
        )

        frontage = pd.to_numeric(frame[LOT_FRONTAGE_COLUMN], errors="coerce")
        if frontage.notna().sum() == 0:
            raise ValueError("Lot Frontage cannot be entirely missing during fit.")

        working = frame.loc[:, [NEIGHBORHOOD_COLUMN, LOT_CONFIG_COLUMN]].copy()
        working[LOT_FRONTAGE_COLUMN] = frontage

        self.neighborhood_medians_ = (
            working.groupby(NEIGHBORHOOD_COLUMN, dropna=False)[LOT_FRONTAGE_COLUMN]
            .median()
            .dropna()
            .to_dict()
        )
        self.lot_config_medians_ = (
            working.groupby(LOT_CONFIG_COLUMN, dropna=False)[LOT_FRONTAGE_COLUMN]
            .median()
            .dropna()
            .to_dict()
        )
        self.global_median_ = float(frontage.median())
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _require_dataframe(X)
        _validate_columns(
            frame,
            (LOT_FRONTAGE_COLUMN, NEIGHBORHOOD_COLUMN, LOT_CONFIG_COLUMN),
            context="Lot Frontage imputer input",
        )

        transformed = frame.copy()
        missing = transformed[LOT_FRONTAGE_COLUMN].isna()

        if not missing.any():
            return transformed

        neighborhood_fill = transformed[NEIGHBORHOOD_COLUMN].map(self.neighborhood_medians_)
        lot_config_fill = transformed[LOT_CONFIG_COLUMN].map(self.lot_config_medians_)

        fill_values = neighborhood_fill.fillna(lot_config_fill).fillna(self.global_median_)
        transformed.loc[missing, LOT_FRONTAGE_COLUMN] = fill_values.loc[missing].astype(float)
        return transformed


class GarageYearImputer(BaseEstimator, TransformerMixin):
    """Handle structural absence and unknown missing Garage Yr Blt values.

    For rows whose training-time representation says there is no garage
    (Garage Type == "NA"), a missing Garage Yr Blt is encoded as 0. Remaining
    missing garage years are filled with the median observed garage year from
    the training partition.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> GarageYearImputer:
        del y
        frame = _require_dataframe(X)
        _validate_columns(
            frame,
            (GARAGE_YEAR_COLUMN, GARAGE_TYPE_COLUMN),
            context="Garage year imputer input",
        )

        garage_year = pd.to_numeric(frame[GARAGE_YEAR_COLUMN], errors="coerce")
        observed_garage = (
            frame[GARAGE_TYPE_COLUMN].ne(STRUCTURAL_ABSENCE_TOKEN)
            & garage_year.notna()
        )

        if not observed_garage.any():
            raise ValueError("No observed garage years are available during fit.")

        self.observed_garage_year_median_ = float(garage_year.loc[observed_garage].median())
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _require_dataframe(X)
        _validate_columns(
            frame,
            (GARAGE_YEAR_COLUMN, GARAGE_TYPE_COLUMN),
            context="Garage year imputer input",
        )

        transformed = frame.copy()
        missing_year = transformed[GARAGE_YEAR_COLUMN].isna()
        no_garage = transformed[GARAGE_TYPE_COLUMN].eq(STRUCTURAL_ABSENCE_TOKEN)

        transformed.loc[missing_year & no_garage, GARAGE_YEAR_COLUMN] = 0.0
        transformed.loc[
            transformed[GARAGE_YEAR_COLUMN].isna(),
            GARAGE_YEAR_COLUMN,
        ] = self.observed_garage_year_median_

        return transformed


def build_primary_preprocessor(*, scale_numeric: bool = True) -> Pipeline:
    """Build the primary leakage-safe preprocessing pipeline.

    The returned object must be fitted on the training feature matrix only.
    Calibration and test features must only call ``transform``.

    Text categorical variables are one-hot encoded in the first baseline,
    including source-documented ordinal text variables. This avoids imposing
    arbitrary equal spacing between ordinal labels. A later controlled
    experiment may compare explicit ordinal encoding against this baseline.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=CATEGORICAL_MISSING_TOKEN,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    columns = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("validate", PrimaryFeatureValidator()),
            ("lot_frontage", HierarchicalLotFrontageImputer()),
            ("garage_year", GarageYearImputer()),
            ("columns", columns),
        ]
    )
