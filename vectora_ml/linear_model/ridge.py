"""
Ridge Regression via batch gradient descent (L2-regularized).
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


class Ridge(BaseEstimator):
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

        self.loss_history_ = []

    def fit(self, X, y):
        """
        Train the ridge regression model via batch gradient descent.
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

            # Same MSE gradient as LinearRegression, plus the L2 penalty's
            # own gradient (2 * alpha * coef_). The intercept has no
            # penalty term, so its update is untouched.
            grad_coef = (2 / n_samples) * (X.T @ error) + 2 * self.alpha * self.coef_
            grad_intercept = (2 / n_samples) * np.sum(error)

            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept

            # Track the actual objective being minimized (MSE + penalty),
            # not just MSE, so the loss curve reflects what gradient
            # descent is really chasing.
            mse = mean_squared_error(y, y_pred)
            penalty = self.alpha * np.sum(self.coef_ ** 2)
            loss = mse + penalty

            if np.isnan(loss) or np.isinf(loss):
                raise ConvergenceError(
                    "Gradient descent diverged (loss is NaN/Inf). "
                    "Try a smaller learning_rate."
                )

            self.loss_history_.append(loss)

            # Same early-stopping rule as LinearRegression: stop once the
            # gradient itself is small, rather than watching loss deltas.
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