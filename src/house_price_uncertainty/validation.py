"""Validation helpers shared by metric and evaluation code."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_1d_float_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return *values* as a finite, non-empty one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def require_same_length(**arrays: NDArray[np.float64]) -> None:
    """Raise when the provided arrays do not all have the same length."""
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Arrays must have equal length; received {details}.")
