"""House-price modeling utilities with uncertainty-aware evaluation."""

from .metrics import (
    empirical_coverage,
    mean_absolute_error,
    mean_interval_width,
    root_mean_squared_error,
)

__all__ = [
    "empirical_coverage",
    "mean_absolute_error",
    "mean_interval_width",
    "root_mean_squared_error",
]

__version__ = "0.1.0"
