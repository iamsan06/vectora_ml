"""
Tests for vectora_ml.linear_model.logistic_regression.LogisticRegression
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from vectora_ml.linear_model.logistic_regression import LogisticRegression
from vectora_ml.core.exceptions import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
    ConvergenceError,
)
from vectora_ml.utils.datasets import make_classification
from vectora_ml.utils.preprocessing import train_test_split
from vectora_ml.core.metrics import accuracy_score


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=300, n_features=2, random_state=0)

    # NOTE: not using StandardScaler here — as of this writing it can't
    # be instantiated (BaseEstimator's abstract `predict` blocks any
    # transformer that doesn't implement one). Standardizing by hand
    # until that's fixed.
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return train_test_split(X, y, test_size=0.2, random_state=0)


# ---------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------

def test_invalid_learning_rate_raises():
    with pytest.raises(InvalidParameterError):
        LogisticRegression(learning_rate=0)

    with pytest.raises(InvalidParameterError):
        LogisticRegression(learning_rate=-0.1)


def test_invalid_max_iter_raises():
    with pytest.raises(InvalidParameterError):
        LogisticRegression(max_iter=0)


def test_invalid_tol_raises():
    with pytest.raises(InvalidParameterError):
        LogisticRegression(tol=0)


def test_invalid_threshold_raises():
    with pytest.raises(InvalidParameterError):
        LogisticRegression(threshold=0)

    with pytest.raises(InvalidParameterError):
        LogisticRegression(threshold=1)

    with pytest.raises(InvalidParameterError):
        LogisticRegression(threshold=1.5)


# ---------------------------------------------------------------------
# Fit / predict behavior
# ---------------------------------------------------------------------

def test_not_fitted_error_before_fit():
    model = LogisticRegression()
    X = np.array([[0.0, 0.0]])

    with pytest.raises(NotFittedError):
        model.predict(X)

    with pytest.raises(NotFittedError):
        model.predict_proba(X)


def test_rejects_non_binary_targets():
    model = LogisticRegression()
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2])  # not binary

    with pytest.raises(InvalidParameterError):
        model.fit(X, y)


def test_fit_sets_expected_attributes(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = LogisticRegression(learning_rate=0.1, max_iter=2000)
    model.fit(X_train, y_train)

    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == X_train.shape[1]
    assert len(model.loss_history_) > 0
    assert model._is_fitted is True


def test_predict_proba_in_valid_range(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = LogisticRegression(learning_rate=0.1, max_iter=2000)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_predict_returns_binary_labels(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = LogisticRegression(learning_rate=0.1, max_iter=2000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert set(np.unique(predictions)).issubset({0, 1})


def test_dimension_mismatch_on_predict(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = LogisticRegression(learning_rate=0.1, max_iter=2000)
    model.fit(X_train, y_train)

    bad_X = X_test[:, :1]  # wrong number of features

    with pytest.raises(DimensionMismatchError):
        model.predict(bad_X)


def test_extreme_learning_rate_stays_finite_due_to_clipping():
    """
    Unlike LinearRegression's MSE, binary_cross_entropy clips predicted
    probabilities into (eps, 1-eps) before taking log(). So even when
    coef_ explodes under a huge learning rate, the loss itself stays
    bounded (never NaN/Inf) — it just won't be a *good* loss. This
    documents that difference rather than expecting a ConvergenceError.
    """

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5)) * 1000  # large scale, no standardization
    y = rng.integers(0, 2, size=50)

    model = LogisticRegression(learning_rate=1e6, max_iter=50)
    model.fit(X, y)  # should not raise

    assert np.isfinite(model.final_loss_)
    assert all(np.isfinite(loss) for loss in model.loss_history_)


def test_loss_history_is_non_increasing_on_average(classification_data):
    """
    Gradient descent won't be monotonic step-to-step, but the loss at
    the end of training should be well below the loss at the start.
    """

    X_train, X_test, y_train, y_test = classification_data

    model = LogisticRegression(learning_rate=0.1, max_iter=2000)
    model.fit(X_train, y_train)

    first_loss = model.loss_history_[0]
    last_loss = model.loss_history_[-1]

    assert last_loss < first_loss


# ---------------------------------------------------------------------
# Correctness vs sklearn
# ---------------------------------------------------------------------

def test_accuracy_comparable_to_sklearn(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = LogisticRegression(learning_rate=0.1, max_iter=5000)
    model.fit(X_train, y_train)
    our_accuracy = model.score(X_test, y_test)

    sklearn_model = SklearnLogisticRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_accuracy = accuracy_score(y_test, sklearn_model.predict(X_test))

    # Same ballpark, not exact match — different solvers/regularization.
    assert abs(our_accuracy - sklearn_accuracy) < 0.1


def test_score_matches_accuracy_score(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = LogisticRegression(learning_rate=0.1, max_iter=2000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    expected = accuracy_score(y_test, predictions)

    assert model.score(X_test, y_test) == pytest.approx(expected)