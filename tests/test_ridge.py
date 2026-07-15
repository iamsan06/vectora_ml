"""
Tests for vectora_ml.linear_model.ridge.Ridge
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge as SklearnRidge

from vectora_ml.linear_model.ridge import Ridge
from vectora_ml.linear_model.linear_regression import LinearRegression
from vectora_ml.core.exceptions import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
    ConvergenceError,
)
from vectora_ml.utils.datasets import make_regression
from vectora_ml.utils.preprocessing import train_test_split


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def regression_data():
    X, y, true_coef = make_regression(
        n_samples=300, n_features=4, noise=1.0, random_state=0
    )

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
        Ridge(alpha=-1.0)


def test_alpha_zero_is_allowed():
    # alpha=0 should behave like plain LinearRegression, not error out.
    model = Ridge(alpha=0.0)
    assert model.alpha == 0.0


def test_invalid_learning_rate_raises():
    with pytest.raises(InvalidParameterError):
        Ridge(learning_rate=0)


def test_invalid_max_iter_raises():
    with pytest.raises(InvalidParameterError):
        Ridge(max_iter=0)


def test_invalid_tol_raises():
    with pytest.raises(InvalidParameterError):
        Ridge(tol=0)


# ---------------------------------------------------------------------
# Fit / predict behavior
# ---------------------------------------------------------------------

def test_not_fitted_error_before_fit():
    model = Ridge()
    X = np.array([[0.0, 0.0]])

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_fit_sets_expected_attributes(regression_data):
    X_train, X_test, y_train, y_test, _ = regression_data

    model = Ridge(alpha=1.0, learning_rate=0.1, max_iter=2000)
    model.fit(X_train, y_train)

    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == X_train.shape[1]
    assert len(model.loss_history_) > 0
    assert model._is_fitted is True


def test_dimension_mismatch_on_predict(regression_data):
    X_train, X_test, y_train, y_test, _ = regression_data

    model = Ridge(alpha=1.0, learning_rate=0.1, max_iter=2000)
    model.fit(X_train, y_train)

    bad_X = X_test[:, :1]

    with pytest.raises(DimensionMismatchError):
        model.predict(bad_X)


def test_diverging_learning_rate_raises_convergence_error(regression_data):
    X_train, X_test, y_train, y_test, _ = regression_data

    model = Ridge(alpha=1.0, learning_rate=1e6, max_iter=50)

    with pytest.raises(ConvergenceError):
        model.fit(X_train, y_train)


def test_loss_decreases_over_training(regression_data):
    X_train, X_test, y_train, y_test, _ = regression_data

    model = Ridge(alpha=1.0, learning_rate=0.1, max_iter=2000)
    model.fit(X_train, y_train)

    assert model.loss_history_[-1] < model.loss_history_[0]


# ---------------------------------------------------------------------
# Regularization behavior — the actual point of Ridge
# ---------------------------------------------------------------------

def test_higher_alpha_shrinks_coefficients_more(regression_data):
    X_train, X_test, y_train, y_test, _ = regression_data

    # NOTE: the L2 penalty's own gradient term is 2*alpha*coef_, so a
    # learning_rate that's stable for small alpha can overshoot and
    # diverge for large alpha (this is a real gradient-descent stability
    # limit, not a bug — the fix is a smaller learning_rate, same as
    # you'd do for any steep loss surface). Use a smaller learning_rate
    # for the high-alpha model so both actually converge.
    low_alpha = Ridge(alpha=0.01, learning_rate=0.05, max_iter=5000)
    high_alpha = Ridge(alpha=2.0, learning_rate=0.02, max_iter=5000)

    low_alpha.fit(X_train, y_train)
    high_alpha.fit(X_train, y_train)

    low_norm = np.sum(low_alpha.coef_ ** 2)
    high_norm = np.sum(high_alpha.coef_ ** 2)

    assert high_norm < low_norm


def test_alpha_zero_approximates_linear_regression(regression_data):
    X_train, X_test, y_train, y_test, _ = regression_data

    ridge = Ridge(alpha=0.0, learning_rate=0.1, max_iter=5000)
    ols = LinearRegression(learning_rate=0.1, max_iter=5000)

    ridge.fit(X_train, y_train)
    ols.fit(X_train, y_train)

    np.testing.assert_allclose(ridge.coef_, ols.coef_, atol=0.05)
    assert ridge.intercept_ == pytest.approx(ols.intercept_, abs=0.05)


# ---------------------------------------------------------------------
# Correctness vs sklearn
# ---------------------------------------------------------------------

def test_r2_comparable_to_sklearn(regression_data):
    X_train, X_test, y_train, y_test, _ = regression_data

    # sklearn's Ridge minimizes sum_squared_residuals + alpha*||w||^2,
    # while ours minimizes MEAN_squared_error + alpha*||w||^2. Since
    # MSE = SSR / n, our alpha is effectively n times stronger than
    # sklearn's for the same value — so to compare like-for-like against
    # sklearn's alpha=1.0, ours needs to be scaled down by n_train.
    sklearn_alpha = 1.0
    n_train = len(y_train)
    our_alpha = sklearn_alpha / n_train

    model = Ridge(alpha=our_alpha, learning_rate=0.1, max_iter=5000)
    model.fit(X_train, y_train)
    our_r2 = model.score(X_test, y_test)

    sklearn_model = SklearnRidge(alpha=sklearn_alpha)
    sklearn_model.fit(X_train, y_train)
    sklearn_r2 = sklearn_model.score(X_test, y_test)

    assert our_r2 > 0.5
    assert abs(our_r2 - sklearn_r2) < 0.05