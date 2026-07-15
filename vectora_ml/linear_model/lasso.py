"""
Lasso Regression via batch gradient descent (L1-regularized).
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
    ConvergenceError,
)
from vectora_ml.core.metrics import mean_squared_error, r2_score
from vectora_ml.utils.validation import check_array, check_X_y


class Lasso(BaseEstimator):
    """
    Linear Regression with L1 regularization, trained with batch
    gradient descent.

    Minimizes: MSE + alpha * sum(|coef_|)

    Unlike Ridge's L2 penalty, which shrinks coefficients smoothly
    toward zero, L1 can push coefficients to *exactly* zero — this
    is what makes Lasso useful for feature selection, not just
    shrinkage. As with Ridge, the intercept is not penalized.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. alpha=0 reduces this to plain
        LinearRegression. Larger alpha zeroes out more coefficients.

    learning_rate : float, default=0.01
        Step size for gradient descent.

    max_iter : int, default=1000
        Maximum number of gradient descent updates.

    tol : float, default=1e-6
        If the gradient's magnitude drops below this value,
        training stops early (the model has converged).
    """

    def __init__(
        self,
        alpha=1.0,
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-6,
    ):
        super().__init__()

        if alpha < 0:
            raise InvalidParameterError("alpha must be non-negative.")

        if learning_rate <= 0:
            raise InvalidParameterError("learning_rate must be positive.")

        if max_iter <= 0:
            raise InvalidParameterError("max_iter must be a positive integer.")

        if tol <= 0:
            raise InvalidParameterError("tol must be positive.")

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        # Not set here on purpose: `_check_is_fitted` uses hasattr(), so
        # setting these to None in __init__ would make the estimator look
        # "fitted" immediately. They only exist after fit() runs.
        self.loss_history_ = []

    def fit(self, X, y):
        """
        Train the lasso regression model via batch gradient descent.
        """

        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.loss_history_ = []

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for i in range(self.max_iter):

            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y

            # |coef_| isn't differentiable at 0, so we use its subgradient
            # instead: sign(coef_), which is -1, 0, or +1. This is a
            # standard, simple way to handle L1 with plain gradient
            # descent — it won't zero out coefficients as cleanly as
            # proper coordinate descent/soft-thresholding would, but it's
            # the same idea and coefficients still shrink toward zero and
            # can land on exactly zero once close enough.
            grad_coef = (2 / n_samples) * (X.T @ error) + self.alpha * np.sign(self.coef_)
            grad_intercept = (2 / n_samples) * np.sum(error)

            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept

            # Track the actual objective being minimized (MSE + penalty),
            # not just MSE, so the loss curve reflects what gradient
            # descent is really chasing.
            mse = mean_squared_error(y, y_pred)
            penalty = self.alpha * np.sum(np.abs(self.coef_))
            loss = mse + penalty

            if np.isnan(loss) or np.isinf(loss):
                raise ConvergenceError(
                    "Gradient descent diverged (loss is NaN/Inf). "
                    "Try a smaller learning_rate."
                )

            self.loss_history_.append(loss)

            # Same early-stopping rule as LinearRegression: stop once the
            # gradient itself is small, rather than watching loss deltas.
            # Note: because of the sign() term, grad_norm can hover near
            # a small non-zero floor instead of decaying to ~0 the way
            # LinearRegression's does — that's expected for L1.
            grad_norm = np.sqrt(
                np.sum(grad_coef ** 2) + grad_intercept ** 2
            )

            if grad_norm < self.tol:
                break

        self.n_iter_ = len(self.loss_history_)
        self.final_loss_ = self.loss_history_[-1]

        self._mark_fitted()

        return self

    def predict(self, X):
        """
        Predict target values for samples in X.
        """

        self._check_is_fitted(["coef_", "intercept_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Return the R^2 score of the model on the given data.
        """

        y_pred = self.predict(X)

        return r2_score(np.asarray(y), y_pred)