"""
Support Vector Classifier (linear, soft-margin) via batch subgradient descent.
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
    ConvergenceError,
)
from vectora_ml.core.metrics import accuracy_score
from vectora_ml.utils.validation import check_array, check_X_y


class SVC(BaseEstimator):
    """
    Linear Support Vector Classifier (soft-margin), trained via batch
    subgradient descent on the hinge loss.

    Minimizes: alpha * ||coef_||^2 + mean(hinge_loss)
    where hinge_loss_i = max(0, 1 - y_i * (coef_ . x_i + intercept_))
    and y is mapped to {-1, +1} internally (same convention as
    AdaBoost, since the margin y_i * decision_i only makes sense with
    signed labels).

    The two terms trade off against each other: alpha * ||coef_||^2
    pushes for a WIDE margin (a decision boundary far from every
    point), while the hinge loss penalizes points that are on the
    wrong side, or too close to, that boundary. A sample correctly
    classified with margin >= 1 contributes nothing to the loss at
    all — this is what makes it "support vector" machine: once a
    point is safely past the margin, gradient descent stops caring
    about it entirely, and only the points near/inside the margin (the
    "support vectors") keep influencing coef_/intercept_.

    Parameters
    ----------
    alpha : float, default=0.01
        Regularization strength — controls margin width vs. how many
        margin violations are tolerated. Same role as Ridge's alpha:
        larger alpha means a wider margin but more tolerance for
        misclassified/close points.

    learning_rate : float, default=0.001
        Step size for gradient descent. Kept small by default since
        the hinge loss's gradient can jump abruptly at the margin.

    max_iter : int, default=1000
        Maximum number of gradient descent updates.

    tol : float, default=1e-6
        If the gradient's magnitude drops below this value, training
        stops early.
    """

    def __init__(
        self,
        alpha=0.01,
        learning_rate=0.001,
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
        Train the SVM via batch subgradient descent on the hinge loss.
        """

        X, y = check_X_y(X, y)

        unique_labels = np.unique(y)

        if not np.all(np.isin(unique_labels, [0, 1])):
            raise InvalidParameterError(
                "SVC only supports binary targets encoded as 0 and 1."
            )

        n_samples, n_features = X.shape

        y_signed = np.where(y == 1, 1, -1)

        self.n_features_in_ = n_features
        self.loss_history_ = []

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for i in range(self.max_iter):

            margins = y_signed * (X @ self.coef_ + self.intercept_)

            # Samples with margin >= 1 are safely classified and
            # correctly outside the margin — hinge loss (and its
            # gradient) is exactly 0 for them. Only "violated" samples
            # (misclassified, or inside the margin) contribute.
            violated = margins < 1

            if np.any(violated):
                hinge_grad_coef = -(X[violated].T @ y_signed[violated]) / n_samples
                hinge_grad_intercept = -np.sum(y_signed[violated]) / n_samples
            else:
                hinge_grad_coef = np.zeros(n_features)
                hinge_grad_intercept = 0.0

            # The L2 term's gradient (2 * alpha * coef_) applies
            # regardless of which samples are violated — same as
            # Ridge. The intercept is never regularized.
            grad_coef = 2 * self.alpha * self.coef_ + hinge_grad_coef
            grad_intercept = hinge_grad_intercept

            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept

            hinge_loss = np.mean(np.maximum(0, 1 - margins))
            loss = self.alpha * np.sum(self.coef_ ** 2) + hinge_loss

            if np.isnan(loss) or np.isinf(loss):
                raise ConvergenceError(
                    "Gradient descent diverged (loss is NaN/Inf). "
                    "Try a smaller learning_rate."
                )

            self.loss_history_.append(loss)

            grad_norm = np.sqrt(
                np.sum(grad_coef ** 2) + grad_intercept ** 2
            )

            if grad_norm < self.tol:
                break

        self.n_iter_ = len(self.loss_history_)
        self.final_loss_ = self.loss_history_[-1]

        self._mark_fitted()

        return self

    def decision_function(self, X):
        """
        Return each sample's signed distance from the decision
        boundary (before thresholding into a class label). Positive
        leans toward class 1, negative toward class 0.
        """

        self._check_is_fitted(["coef_", "intercept_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """
        Predict class labels (0 or 1) via the sign of the decision
        function.
        """

        return (self.decision_function(X) >= 0).astype(int)

    def score(self, X, y):
        """
        Return the classification accuracy of the model on the given data.
        """

        y_pred = self.predict(X)

        return accuracy_score(np.asarray(y), y_pred)