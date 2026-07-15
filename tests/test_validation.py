import numpy as np
import pytest
from vectora_ml.utils.validation import (
    check_array,
    check_X_y,
    check_is_fitted,
)
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
)

class DummyEstimator:
    pass

def test_check_array_valid():
    X = [[1, 2], [3, 4]]
    result = check_array(X)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

def test_check_array_invalid_dimension():
    X = [1, 2, 3]

    with pytest.raises(DimensionMismatchError):
        check_array(X)

def test_check_array_nan():
    X = [[1, np.nan]]

    with pytest.raises(InvalidParameterError):
        check_array(X)

def test_check_X_y_valid():

    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])

    X_checked, y_checked = check_X_y(X, y)

    assert X_checked.shape == (3, 1)
    assert y_checked.shape == (3,)

def test_check_X_y_length_mismatch():

    X = np.array([[1], [2]])
    y = np.array([1])

    with pytest.raises(DimensionMismatchError):
        check_X_y(X, y)

def test_check_X_y_invalid_y_dimension():

    X = np.array([[1], [2]])

    y = np.array([[1], [2]])

    with pytest.raises(DimensionMismatchError):
        check_X_y(X, y)

def test_check_is_fitted():

    estimator = DummyEstimator()

    with pytest.raises(NotFittedError):
        check_is_fitted(estimator, "coef_")

def test_check_is_fitted_success():

    estimator = DummyEstimator()

    estimator.coef_ = np.array([1.0])

    check_is_fitted(estimator, "coef_")