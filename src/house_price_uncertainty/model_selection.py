"""Model-selection utilities restricted to the training partition."""

from __future__ import annotations

from sklearn.model_selection import KFold

TRAINING_CV_FOLDS = 5
TRAINING_CV_RANDOM_STATE = 2026


def make_training_cv(
    *,
    n_splits: int = TRAINING_CV_FOLDS,
    random_state: int = TRAINING_CV_RANDOM_STATE,
) -> KFold:
    """Create the deterministic CV protocol used inside training data."""
    if n_splits < 2:
        raise ValueError("Training cross-validation requires at least two folds.")

    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )