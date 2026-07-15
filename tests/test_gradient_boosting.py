"""
Tests for vectora_ml.ensemble.gradient_boosting.GradientBoostingRegressor
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression as sk_make_regression
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.ensemble import GradientBoostingRegressor as SklearnGradientBoostingRegressor

from vectora_ml.ensemble.gradient_boosting import GradientBoostingRegressor
from vectora_ml.core.exceptions import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def regression_data():
    X, y = sk_make_regression(
        n_samples=400, n_features=8, noise=10.0, random_state=0
    )
    return sk_train_test_split(X, y, test_size=0.2, random_state=0)


# ---------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------

def test_invalid_n_estimators_raises():
    with pytest.raises(InvalidParameterError):
        GradientBoostingRegressor(n_estimators=0)


def test_invalid_learning_rate_raises():
    with pytest.raises(InvalidParameterError):
        GradientBoostingRegressor(learning_rate=0)


def test_invalid_max_depth_raises():
    with pytest.raises(InvalidParameterError):
        GradientBoostingRegressor(max_depth=0)


def test_invalid_min_samples_split_raises():
    with pytest.raises(InvalidParameterError):
        GradientBoostingRegressor(min_samples_split=1)


# ---------------------------------------------------------------------
# Fit / predict behavior
# ---------------------------------------------------------------------

def test_not_fitted_error_before_fit():
    model = GradientBoostingRegressor()
    X = np.array([[0.0]])

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_fit_sets_expected_attributes(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    model = GradientBoostingRegressor(n_estimators=20, max_depth=3)
    model.fit(X_train, y_train)

    assert len(model.estimators_) == 20
    assert len(model.staged_loss_) == 20
    assert hasattr(model, "initial_prediction_")
    assert model._is_fitted is True


def test_initial_prediction_is_training_mean(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    model = GradientBoostingRegressor(n_estimators=5)
    model.fit(X_train, y_train)

    assert model.initial_prediction_ == pytest.approx(np.mean(y_train))


def test_dimension_mismatch_on_predict(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    model = GradientBoostingRegressor(n_estimators=10)
    model.fit(X_train, y_train)

    with pytest.raises(DimensionMismatchError):
        model.predict(X_test[:, :3])


def test_staged_loss_decreases_over_rounds(regression_data):
    """
    Each round fits a tree to the CURRENT residuals — training MSE
    should trend downward as more rounds are added (not necessarily
    monotonically every single round with a high learning rate, but
    clearly lower at the end than at the start).
    """

    X_train, X_test, y_train, y_test = regression_data

    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    assert model.staged_loss_[-1] < model.staged_loss_[0]


def test_more_estimators_improves_training_fit(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    few = GradientBoostingRegressor(n_estimators=5, learning_rate=0.1, max_depth=3)
    many = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

    few.fit(X_train, y_train)
    many.fit(X_train, y_train)

    assert many.score(X_train, y_train) > few.score(X_train, y_train)


def test_smaller_learning_rate_needs_more_rounds_for_same_fit(regression_data):
    """
    This is the shrinkage trade-off the docstring describes: a small
    learning_rate with few rounds should underfit relative to a larger
    learning_rate with the same number of rounds.
    """

    X_train, X_test, y_train, y_test = regression_data

    small_lr = GradientBoostingRegressor(n_estimators=10, learning_rate=0.01, max_depth=3)
    large_lr = GradientBoostingRegressor(n_estimators=10, learning_rate=0.5, max_depth=3)

    small_lr.fit(X_train, y_train)
    large_lr.fit(X_train, y_train)

    assert large_lr.score(X_train, y_train) > small_lr.score(X_train, y_train)


# ---------------------------------------------------------------------
# Correctness vs sklearn
# ---------------------------------------------------------------------

def test_r2_comparable_to_sklearn(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    our_r2 = model.score(X_test, y_test)

    sklearn_model = SklearnGradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_r2 = sklearn_model.score(X_test, y_test)

    assert our_r2 > 0.5
    assert abs(our_r2 - sklearn_r2) < 0.1