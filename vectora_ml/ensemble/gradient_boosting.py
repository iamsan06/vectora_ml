"""
Gradient Boosting Regressor — sequential trees fit to residuals.
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)
from vectora_ml.core.metrics import mean_squared_error, r2_score
from vectora_ml.utils.validation import check_array, check_X_y
from vectora_ml.tree.decision_tree_regressor import DecisionTreeRegressor


class GradientBoostingRegressor(BaseEstimator):
    """
    Gradient Boosting for regression, trained under squared-error loss.

    Starts from a constant prediction (the training mean) and
    repeatedly fits a shallow regression tree to the CURRENT residuals
    (actual - predicted-so-far). Each tree's output is added to the
    running prediction, shrunk by `learning_rate`, and residuals are
    recomputed for the next round.

    Why "residuals" and why this counts as "gradient" boosting: for
    squared-error loss L = (y - F)^2, the negative gradient of L with
    respect to the current prediction F is exactly (y - F) — the
    residual. So "fit a tree to the residuals" IS gradient descent —
    just taken in function space (each tree is one step) rather than
    parameter space (like the weight updates in LinearRegression).
    Every boosting round nudges F(x) a little further downhill.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds (trees).

    learning_rate : float, default=0.1
        Shrinks each tree's contribution before adding it to the
        running prediction. Smaller values need more estimators but
        tend to generalize better — small, many steps beats few, large
        ones, for the same reason it does in plain gradient descent.

    max_depth : int, default=3
        Depth of each tree. Kept shallow ("weak learners") on purpose:
        the boosting process as a whole fits the data, not any single
        tree — deep trees here would let one round overfit and defeat
        the point of the ensemble.

    min_samples_split : int, default=2
        Minimum number of samples a tree node must have to be split.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
    ):
        super().__init__()

        if n_estimators <= 0:
            raise InvalidParameterError(
                "n_estimators must be a positive integer."
            )

        if learning_rate <= 0:
            raise InvalidParameterError("learning_rate must be positive.")

        if max_depth <= 0:
            raise InvalidParameterError(
                "max_depth must be a positive integer."
            )

        if min_samples_split < 2:
            raise InvalidParameterError(
                "min_samples_split must be at least 2."
            )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        """
        Train the ensemble: start from the mean, then repeatedly fit a
        tree to the current residuals and add its shrunk prediction to
        the running total.
        """

        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]

        # Round 0: under squared-error loss, the best possible constant
        # prediction is the mean — this is where boosting starts from.
        self.initial_prediction_ = np.mean(y)

        current_predictions = np.full(y.shape, self.initial_prediction_)

        self.estimators_ = []
        self.staged_loss_ = []

        for m in range(self.n_estimators):

            residuals = y - current_predictions

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X, residuals)

            update = tree.predict(X)
            current_predictions = current_predictions + self.learning_rate * update

            self.estimators_.append(tree)
            self.staged_loss_.append(mean_squared_error(y, current_predictions))

        self.n_iter_ = len(self.estimators_)
        self.final_loss_ = self.staged_loss_[-1]

        self._mark_fitted()

        return self

    def predict(self, X):
        """
        Predict by summing the initial prediction and every tree's
        shrunk contribution.
        """

        self._check_is_fitted(
            ["estimators_", "initial_prediction_", "n_features_in_"]
        )

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        predictions = np.full(X.shape[0], self.initial_prediction_)

        for tree in self.estimators_:
            predictions = predictions + self.learning_rate * tree.predict(X)

        return predictions

    def score(self, X, y):
        """
        Return the R^2 score of the ensemble on the given data.
        """

        y_pred = self.predict(X)

        return r2_score(np.asarray(y), y_pred)