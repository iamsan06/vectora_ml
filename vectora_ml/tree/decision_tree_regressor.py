"""
Decision Tree Regressor via recursive binary splitting (variance reduction).
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)
from vectora_ml.core.metrics import r2_score
from vectora_ml.utils.validation import check_array, check_X_y
from vectora_ml.tree.node import Node


class DecisionTreeRegressor(BaseEstimator):
    """
    Decision Tree Regressor trained via recursive binary splitting.

    Same recursive splitting and stopping logic as
    DecisionTreeClassifier, but splits are chosen to minimize the
    weighted variance of the target in the two children rather than
    impurity over class labels. Leaves predict the mean target value
    of the samples that reached them.

    After fitting, `feature_importances_` reports how much each
    feature contributed to reducing variance across the whole tree.

    Primarily exists as the weak learner used inside
    GradientBoostingRegressor.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree.

    min_samples_split : int, default=2
        Minimum number of samples a node must have to be split further.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        super().__init__()

        if max_depth is not None and max_depth <= 0:
            raise InvalidParameterError(
                "max_depth must be a positive integer or None."
            )

        if min_samples_split < 2:
            raise InvalidParameterError(
                "min_samples_split must be at least 2."
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    @staticmethod
    def _variance(y):
        if len(y) == 0:
            return 0.0

        return np.var(y)

    def _best_split(self, X, y):
        """
        Returns None if no split reduces variance, otherwise
        (feature_index, threshold, gain).
        """

        n_samples, n_features = X.shape
        parent_variance = self._variance(y)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):

            values = np.unique(X[:, feature_index])

            if len(values) < 2:
                continue

            candidate_thresholds = (values[:-1] + values[1:]) / 2

            for threshold in candidate_thresholds:

                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                weighted_variance = (
                    (len(y_left) / n_samples) * self._variance(y_left)
                    + (len(y_right) / n_samples) * self._variance(y_right)
                )

                gain = parent_variance - weighted_variance

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        if best_feature is None:
            return None

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or self._variance(y) == 0
        ):
            return Node(value=np.mean(y))

        split = self._best_split(X, y)

        if split is None:
            return Node(value=np.mean(y))

        feature_index, threshold, gain = split

        # Feature importance: variance reduction from this split,
        # weighted by the fraction of total training samples that
        # passed through this node.
        self._raw_importances[feature_index] += (
            (n_samples / self._root_n_samples) * gain
        )

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left,
            right=right,
        )

    def fit(self, X, y):
        """
        Train the regression tree by recursively splitting on the
        feature/threshold pair that most reduces variance at each
        node. Also computes `feature_importances_`.
        """

        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]

        self._root_n_samples = X.shape[0]
        self._raw_importances = np.zeros(self.n_features_in_)

        self.root_ = self._build_tree(X, y, depth=0)

        total = np.sum(self._raw_importances)

        if total > 0:
            self.feature_importances_ = self._raw_importances / total
        else:
            self.feature_importances_ = self._raw_importances.copy()

        self._mark_fitted()

        return self

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)

        return self._traverse(x, node.right)

    def predict(self, X):
        """
        Predict target values for samples in X.
        """

        self._check_is_fitted(["root_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        return np.array([self._traverse(row, self.root_) for row in X])

    def score(self, X, y):
        """
        Return the R^2 score of the model on the given data.
        """

        y_pred = self.predict(X)

        return r2_score(np.asarray(y), y_pred)