"""
Tests for vectora_ml.tree.decision_tree_regressor.DecisionTreeRegressor
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression as sk_make_regression
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

from vectora_ml.tree.decision_tree_regressor import DecisionTreeRegressor
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
        n_samples=300, n_features=5, noise=5.0, random_state=0
    )
    return sk_train_test_split(X, y, test_size=0.2, random_state=0)


# ---------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------

def test_invalid_max_depth_raises():
    with pytest.raises(InvalidParameterError):
        DecisionTreeRegressor(max_depth=0)


def test_invalid_min_samples_split_raises():
    with pytest.raises(InvalidParameterError):
        DecisionTreeRegressor(min_samples_split=1)


# ---------------------------------------------------------------------
# Fit / predict behavior
# ---------------------------------------------------------------------

def test_not_fitted_error_before_fit():
    model = DecisionTreeRegressor()
    X = np.array([[0.0]])

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_fit_sets_expected_attributes(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)

    assert hasattr(model, "root_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == X_train.shape[1]
    assert model._is_fitted is True


def test_dimension_mismatch_on_predict(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)

    with pytest.raises(DimensionMismatchError):
        model.predict(X_test[:, :2])


def test_leaf_predicts_mean_of_training_targets():
    """
    With min_samples_split set higher than the dataset size, the tree
    can't split at all — the single leaf should predict exactly the
    mean of y, since that's the value that minimizes squared error
    for a constant prediction.
    """

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])

    model = DecisionTreeRegressor(min_samples_split=100)
    model.fit(X, y)

    predictions = model.predict(X)

    assert np.allclose(predictions, np.mean(y))


def test_unconstrained_tree_fits_training_data_closely():
    """
    An unconstrained tree should be able to memorize a small, clean
    dataset almost exactly.
    """

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 10.0, 11.0, 20.0])

    model = DecisionTreeRegressor()
    model.fit(X, y)

    predictions = model.predict(X)

    assert np.allclose(predictions, y, atol=1e-6)


def test_deeper_tree_reduces_training_error(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    shallow = DecisionTreeRegressor(max_depth=1)
    deep = DecisionTreeRegressor(max_depth=10)

    shallow.fit(X_train, y_train)
    deep.fit(X_train, y_train)

    # Deeper trees fit training data more closely (can overfit more) —
    # R^2 on the TRAINING set should reflect that.
    assert deep.score(X_train, y_train) > shallow.score(X_train, y_train)


# ---------------------------------------------------------------------
# Correctness vs sklearn
# ---------------------------------------------------------------------

def test_r2_comparable_to_sklearn(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)
    our_r2 = model.score(X_test, y_test)

    sklearn_model = SklearnDecisionTreeRegressor(max_depth=5, random_state=0)
    sklearn_model.fit(X_train, y_train)
    sklearn_r2 = sklearn_model.score(X_test, y_test)

    assert abs(our_r2 - sklearn_r2) < 0.15