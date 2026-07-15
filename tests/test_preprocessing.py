import numpy as np
import pytest

from vectora_ml.utils.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)

from vectora_ml.core.exceptions import NotFittedError


def test_standard_scaler():

    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled.mean(axis=0), 0.0)
    assert np.allclose(X_scaled.std(axis=0), 1.0)


def test_standard_inverse():

    X = np.random.rand(10, 4)

    scaler = StandardScaler()

    recovered = scaler.inverse_transform(
        scaler.fit_transform(X)
    )

    assert np.allclose(X, recovered)


def test_standard_not_fitted():

    scaler = StandardScaler()

    with pytest.raises(NotFittedError):
        scaler.transform([[1, 2]])


def test_minmax_scaler():

    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled.min(axis=0), 0)
    assert np.allclose(X_scaled.max(axis=0), 1)


def test_minmax_inverse():

    X = np.random.rand(20, 5)

    scaler = MinMaxScaler()

    recovered = scaler.inverse_transform(
        scaler.fit_transform(X)
    )

    assert np.allclose(X, recovered)


def test_minmax_not_fitted():

    scaler = MinMaxScaler()

    with pytest.raises(NotFittedError):
        scaler.transform([[1, 2]])