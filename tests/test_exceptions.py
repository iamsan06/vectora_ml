import pytest
from vectora_ml.core.exceptions import (
    VectoraMLError,
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
    ConvergenceError,
)

def test_not_fitted_error():
    with pytest.raises(NotFittedError):
        raise NotFittedError("Model has not been fitted.")

def test_dimension_mismatch_error():
    with pytest.raises(DimensionMismatchError):
        raise DimensionMismatchError("Dimension mismatch.")

def test_invalid_parameter_error():
    with pytest.raises(InvalidParameterError):
        raise InvalidParameterError("Invalid parameter.")

def test_convergence_error():
    with pytest.raises(ConvergenceError):
        raise ConvergenceError("Did not converge.")

def test_all_exceptions_inherit_base():
    assert issubclass(NotFittedError, VectoraMLError)
    assert issubclass(DimensionMismatchError, VectoraMLError)
    assert issubclass(InvalidParameterError, VectoraMLError)
    assert issubclass(ConvergenceError, VectoraMLError)