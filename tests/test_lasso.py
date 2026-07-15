"""
Tests for vectora_ml.linear_model.lasso.Lasso
"""

import numpy as np
import pytest
from sklearn.linear_model import Lasso as SklearnLasso

from vectora_ml.linear_model.lasso import Lasso
from vectora_ml.linear_model.linear_regression import LinearRegression
from vectora_ml.core.exceptions import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
    ConvergenceError,
)
from vectora_ml.utils.preprocessing import train_test_split


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sparse_regression_data():
    """
    A regression dataset where only the first 2 of 5 features actually
    influence y — the other 3 are pure noise. This is the setup that
    actually exercises what makes Lasso different from Ridge: it should
    push the 3 irrelevant coefficients toward zero.
    """

    rng = np.random.default_rng(0)

    n_samples, n_features = 300, 5
    X = rng.normal(size=(n_samples, n_features))

    true_coef = np.array([3.0, -2.0, 0.0, 0.0, 0.0])
    y = X @ true_coef + 0.5 + rng.normal(scale=0.3, size=n_samples)

    # NOTE: not using StandardScaler here — as of this writing it can't
    # be instantiated (BaseEstimator's abstract `predict` blocks any
    # transformer that doesn't implement one). Standardizing by hand
    # until that's fixed.
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    return X_train, X_test, y_train, y_test, true_coef


# ---------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------

def test_invalid_alpha_raises():
    with pytest.raises(InvalidParameterError):
        Lasso(alpha=-1.0)


def test_alpha_zero_is_allowed():
    model = Lasso(alpha=0.0)
    assert model.alpha == 0.0


def test_invalid_learning_rate_raises():
    with pytest.raises(InvalidParameterError):
        Lasso(learning_rate=0)


def test_invalid_max_iter_raises():
    with pytest.raises(InvalidParameterError):
        Lasso(max_iter=0)


def test_invalid_tol_raises():
    with pytest.raises(InvalidParameterError):
        Lasso(tol=0)


# ---------------------------------------------------------------------
# Fit / predict behavior
# ---------------------------------------------------------------------

def test_not_fitted_error_before_fit():
    model = Lasso()
    X = np.array([[0.0, 0.0]])

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_fit_sets_expected_attributes(sparse_regression_data):
    X_train, X_test, y_train, y_test, _ = sparse_regression_data

    model = Lasso(alpha=0.1, learning_rate=0.05, max_iter=3000)
    model.fit(X_train, y_train)

    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == X_train.shape[1]
    assert len(model.loss_history_) > 0
    assert model._is_fitted is True


def test_dimension_mismatch_on_predict(sparse_regression_data):
    X_train, X_test, y_train, y_test, _ = sparse_regression_data

    model = Lasso(alpha=0.1, learning_rate=0.05, max_iter=3000)
    model.fit(X_train, y_train)

    bad_X = X_test[:, :2]

    with pytest.raises(DimensionMismatchError):
        model.predict(bad_X)


def test_diverging_learning_rate_raises_convergence_error(sparse_regression_data):
    X_train, X_test, y_train, y_test, _ = sparse_regression_data

    model = Lasso(alpha=0.1, learning_rate=1e6, max_iter=50)

    with pytest.raises(ConvergenceError):
        model.fit(X_train, y_train)


def test_loss_decreases_over_training(sparse_regression_data):
    X_train, X_test, y_train, y_test, _ = sparse_regression_data

    model = Lasso(alpha=0.1, learning_rate=0.05, max_iter=3000)
    model.fit(X_train, y_train)

    assert model.loss_history_[-1] < model.loss_history_[0]


# ---------------------------------------------------------------------
# Regularization behavior — the actual point of Lasso
# ---------------------------------------------------------------------

def test_irrelevant_coefficients_shrink_toward_zero(sparse_regression_data):
    X_train, X_test, y_train, y_test, true_coef = sparse_regression_data

    model = Lasso(alpha=0.3, learning_rate=0.05, max_iter=5000)
    model.fit(X_train, y_train)

    relevant_idx = np.where(true_coef != 0)[0]
    irrelevant_idx = np.where(true_coef == 0)[0]

    relevant_magnitude = np.mean(np.abs(model.coef_[relevant_idx]))
    irrelevant_magnitude = np.mean(np.abs(model.coef_[irrelevant_idx]))

    # The irrelevant features' coefficients should end up much smaller
    # than the relevant ones' — this is Lasso's defining behavior.
    assert irrelevant_magnitude < relevant_magnitude
    assert irrelevant_magnitude < 0.3


def test_higher_alpha_shrinks_coefficients_more(sparse_regression_data):
    X_train, X_test, y_train, y_test, _ = sparse_regression_data

    low_alpha = Lasso(alpha=0.01, learning_rate=0.05, max_iter=5000)
    high_alpha = Lasso(alpha=1.0, learning_rate=0.05, max_iter=5000)

    low_alpha.fit(X_train, y_train)
    high_alpha.fit(X_train, y_train)

    low_norm = np.sum(np.abs(low_alpha.coef_))
    high_norm = np.sum(np.abs(high_alpha.coef_))

    assert high_norm < low_norm


def test_alpha_zero_approximates_linear_regression(sparse_regression_data):
    X_train, X_test, y_train, y_test, _ = sparse_regression_data

    lasso = Lasso(alpha=0.0, learning_rate=0.05, max_iter=5000)
    ols = LinearRegression(learning_rate=0.05, max_iter=5000)

    lasso.fit(X_train, y_train)
    ols.fit(X_train, y_train)

    np.testing.assert_allclose(lasso.coef_, ols.coef_, atol=0.1)


# ---------------------------------------------------------------------
# Correctness vs sklearn
# ---------------------------------------------------------------------

def test_r2_comparable_to_sklearn(sparse_regression_data):
    X_train, X_test, y_train, y_test, _ = sparse_regression_data

    model = Lasso(alpha=0.1, learning_rate=0.05, max_iter=5000)
    model.fit(X_train, y_train)
    our_r2 = model.score(X_test, y_test)

    # Different alpha scaling/solver from sklearn (coordinate descent
    # with soft-thresholding vs. our subgradient descent), so we just
    # check both land in a reasonable, similar R^2 range.
    sklearn_model = SklearnLasso(alpha=0.1)
    sklearn_model.fit(X_train, y_train)
    sklearn_r2 = sklearn_model.score(X_test, y_test)

    assert our_r2 > 0.5
    assert abs(our_r2 - sklearn_r2) < 0.2