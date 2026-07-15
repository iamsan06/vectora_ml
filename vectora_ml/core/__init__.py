"""
Core module exports.
"""

from .metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .exceptions import (
    VectoraMLError,
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
    ConvergenceError,
)

from .estimator import BaseEstimator

__all__ = [
    "BaseEstimator",
    "VectoraMLError",
    "NotFittedError",
    "DimensionMismatchError",
    "InvalidParameterError",
    "ConvergenceError",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
]