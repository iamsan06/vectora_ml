"""
Input validation utilities.

These functions ensure that all estimators receive valid,
consistent NumPy arrays before computation begins.
"""

import numpy as np

from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)
# Re-exported for backward compatibility — the real implementation
# now lives in core.validation, since core.estimator depends on it
# directly and core must never import from utils.
from vectora_ml.core.validation import check_is_fitted


def check_array(X, ensure_2d=True, dtype=float):
    """
    Validate an input feature matrix.

    Parameters
    ----------
    X : array-like

    ensure_2d : bool
        Whether the array must be two-dimensional.

    dtype : type
        Desired NumPy dtype.

    Returns
    -------
    ndarray
    """

    X = np.asarray(X, dtype=dtype)

    if ensure_2d and X.ndim != 2:
        raise DimensionMismatchError(
            f"Expected 2D array but got {X.ndim}D array."
        )

    if np.isnan(X).any():
        raise InvalidParameterError(
            "Input contains NaN values."
        )

    return X


def check_X_y(X, y):
    """
    Validate training data.

    Returns validated X and y.
    """

    X = check_array(X)

    y = np.asarray(y)

    if y.ndim != 1:
        raise DimensionMismatchError(
            "Target vector y must be one-dimensional."
        )

    if len(X) != len(y):
        raise DimensionMismatchError(
            f"X contains {len(X)} samples but y contains {len(y)} samples."
        )

    if np.isnan(y).any():
        raise InvalidParameterError(
            "Target contains NaN values."
        )

    return X, y