"""
Logistic Regression via batch gradient descent.
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
    ConvergenceError,
)
from vectora_ml.core.metrics import binary_cross_entropy, accuracy_score
from vectora_ml.utils.validation import check_array, check_X_y


class LogisticRegression(BaseEstimator):
    """
    Binary Logistic Regression trained with batch gradient descent.

    Models P(y=1 | X) as sigmoid(X @ coef_ + intercept_), and learns
    coef_/intercept_ by minimizing binary cross-entropy loss.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent.

    max_iter : int, default=1000
        Maximum number of gradient descent updates.

    tol : float, default=1e-6
        If the gradient's magnitude drops below this value,
        training stops early (the model has converged).

    threshold : float, default=0.5
        Probability cutoff used by predict() to assign the
        positive class.
    """

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-6,
        threshold=0.5,
    ):
        super().__init__()

        if learning_rate <= 0:
            raise InvalidParameterError("learning_rate must be positive.")

        if max_iter <= 0:
            raise InvalidParameterError("max_iter must be a positive integer.")

        if tol <= 0:
            raise InvalidParameterError("tol must be positive.")

        if not 0 < threshold < 1:
            raise InvalidParameterError("threshold must be between 0 and 1.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold

        # Not set here on purpose: `_check_is_fitted` uses hasattr(), so
        # setting these to None in __init__ would make the estimator look
        # "fitted" immediately. They only exist after fit() runs.
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        """
        Numerically stable sigmoid: 1 / (1 + e^-z).

        Splits on the sign of z so exp() is never called on a large
        positive number, which would overflow. This does NOT change
        the math, only how it's computed.
        """

        result = np.empty_like(z, dtype=float)

        positive = z >= 0
        negative = ~positive

        result[positive] = 1 / (1 + np.exp(-z[positive]))

        exp_z = np.exp(z[negative])
        result[negative] = exp_z / (1 + exp_z)

        return result

    def fit(self, X, y):
        """
        Train the logistic regression model via batch gradient descent.
        """

        X, y = check_X_y(X, y)

        unique_labels = np.unique(y)

        if not np.all(np.isin(unique_labels, [0, 1])):
            raise InvalidParameterError(
                "LogisticRegression only supports binary targets "
                "encoded as 0 and 1."
            )

        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.loss_history_ = []

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for i in range(self.max_iter):

            z = X @ self.coef_ + self.intercept_
            y_pred = self._sigmoid(z)
            error = y_pred - y

            # Same form as linear regression's gradient (2/n * X.T @ error),
            # except here error comes from sigmoid(z) - y instead of
            # (z - y). That's the one place the "theory" shows up: BCE's
            # gradient w.r.t. z collapses to this same clean expression,
            # which is why logistic regression is just gradient descent
            # on a squashed linear model.
            grad_coef = (1 / n_samples) * (X.T @ error)
            grad_intercept = (1 / n_samples) * np.sum(error)

            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept

            loss = binary_cross_entropy(y, y_pred)

            if np.isnan(loss) or np.isinf(loss):
                raise ConvergenceError(
                    "Gradient descent diverged (loss is NaN/Inf). "
                    "Try a smaller learning_rate."
                )

            self.loss_history_.append(loss)

            # Same early-stopping rule as LinearRegression: stop once the
            # gradient itself is small, rather than watching loss deltas,
            # since BCE can plateau on flat regions before coef_ actually
            # converges.
            grad_norm = np.sqrt(
                np.sum(grad_coef ** 2) + grad_intercept ** 2
            )

            if grad_norm < self.tol:
                break

        self.n_iter_ = len(self.loss_history_)
        self.final_loss_ = self.loss_history_[-1]

        self._mark_fitted()

        return self

    def predict_proba(self, X):
        """
        Predict P(y=1 | X) for samples in X.
        """

        self._check_is_fitted(["coef_", "intercept_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        z = X @ self.coef_ + self.intercept_

        return self._sigmoid(z)

    def predict(self, X):
        """
        Predict binary class labels (0 or 1) for samples in X.
        """

        probabilities = self.predict_proba(X)
        

        return (probabilities >= self.threshold).astype(int)

    def score(self, X, y):
        """
        Return the classification accuracy of the model on the given data.
        """

        y_pred = self.predict(X)

        return accuracy_score(np.asarray(y), y_pred)