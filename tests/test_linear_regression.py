"""
Tests for LinearRegression.
"""
import numpy as np
import pytest
from vectora_ml.linear_model import LinearRegression
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
)

def test_fit():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
    ])

    y = np.array([
        2,
        4,
        6,
        8,
    ])

    model = LinearRegression(
        learning_rate=0.01,
        max_iter=5000,
    )

    model.fit(X, y)

    assert model.coef_.shape == (1,)
    assert np.isclose(model.coef_[0], 2.0, atol=1e-2)
    assert np.isclose(model.intercept_, 0.0, atol=1e-2)


def test_predict():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
    ])

    y = np.array([
        2,
        4,
        6,
        8,
    ])

    model = LinearRegression(
        learning_rate=0.01,
        max_iter=5000,
    )

    model.fit(X, y)

    pred = model.predict(np.array([[5]]))

    assert np.isclose(pred[0], 10.0, atol=1e-2)


def test_score():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
    ])

    y = np.array([
        2,
        4,
        6,
        8,
    ])

    model = LinearRegression(
        learning_rate=0.01,
        max_iter=5000,
    )

    model.fit(X, y)

    assert np.isclose(
        model.score(X, y),
        1.0,
        atol=1e-6,
    )


def test_predict_before_fit():
    model = LinearRegression()

    with pytest.raises(NotFittedError):
        model.predict(np.array([[1]]))


def test_dimension_mismatch():
    X = np.array([
        [1],
        [2],
        [3],
    ])

    y = np.array([
        2,
        4,
        6,
    ])

    model = LinearRegression()

    model.fit(X, y)

    with pytest.raises(DimensionMismatchError):
        model.predict(
            np.array([
                [1, 2],
            ])
        )


def test_invalid_learning_rate():
    with pytest.raises(InvalidParameterError):
        LinearRegression(
            learning_rate=0,
        )


def test_invalid_max_iter():
    with pytest.raises(InvalidParameterError):
        LinearRegression(
            max_iter=0,
        )


def test_invalid_tol():
    with pytest.raises(InvalidParameterError):
        LinearRegression(
            tol=-1,
        )


def test_loss_decreases():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
    ])

    y = np.array([
        2,
        4,
        6,
        8,
    ])

    model = LinearRegression(
        learning_rate=0.01,
        max_iter=100,
    )

    model.fit(X, y)

    assert model.loss_history_[0] > model.loss_history_[-1]


def test_training_attributes():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
    ])

    y = np.array([
        2,
        4,
        6,
        8,
    ])

    model = LinearRegression()

    model.fit(X, y)

    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "loss_history_")
    assert hasattr(model, "n_iter_")
    assert hasattr(model, "final_loss_")
    assert hasattr(model, "n_features_in_")

    assert model.n_iter_ > 0
    assert model.final_loss_ >= 0