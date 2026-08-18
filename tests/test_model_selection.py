"""Tests for the frozen training cross-validation protocol."""

from __future__ import annotations

import numpy as np

from house_price_uncertainty.model_selection import make_training_cv


def test_training_cv_is_deterministic() -> None:
    rows = np.arange(30)

    first = list(make_training_cv().split(rows))
    second = list(make_training_cv().split(rows))

    assert len(first) == len(second)

    for (first_train, first_valid), (second_train, second_valid) in zip(
        first,
        second,
        strict=True,
    ):
        assert np.array_equal(first_train, second_train)
        assert np.array_equal(first_valid, second_valid)


def test_training_cv_validation_folds_cover_every_row_once() -> None:
    rows = np.arange(30)

    validation_indices = np.concatenate(
        [
            validation
            for _, validation in make_training_cv().split(rows)
        ]
    )

    assert sorted(validation_indices.tolist()) == rows.tolist()


def test_training_cv_has_no_train_validation_overlap() -> None:
    rows = np.arange(30)

    for train, validation in make_training_cv().split(rows):
        assert set(train).isdisjoint(validation)