"""
Random Forest Classifier via bootstrap aggregation of Decision Trees.
"""

from collections import Counter

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)
from vectora_ml.core.metrics import accuracy_score
from vectora_ml.utils.validation import check_array, check_X_y
from vectora_ml.tree.decision_tree import DecisionTreeClassifier


class RandomForestClassifier(BaseEstimator):
    """
    Random Forest Classifier: an ensemble of Decision Trees, each
    trained on a bootstrap sample of the data, that vote on the
    final prediction.

    Two sources of randomness decorrelate the trees from each other,
    which is what makes averaging them help:
      1. Bagging — each tree sees a different bootstrap sample (drawn
         with replacement) rather than the full training set.
      2. Per-split feature subsampling — each tree only considers
         `max_features` features at each split, handled internally by
         DecisionTreeClassifier. Without this, trees trained on similar
         bootstrap samples tend to pick the same strong feature at the
         root every time and end up highly correlated with each other,
         which defeats the point of averaging.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

    max_depth : int or None, default=None
        Maximum depth of each tree.

    min_samples_split : int, default=2
        Minimum number of samples a node must have to be split further.

    criterion : {"gini", "entropy"}, default="gini"
        Impurity measure used by each tree.

    max_features : None, "sqrt", "log2", int, or float, default="sqrt"
        Number of features each tree considers per split. Defaults to
        "sqrt", the standard Random Forest choice — with max_features=
        None every tree would consider all features at every split,
        which makes this bagging (bagged trees) rather than a proper
        random forest.

    random_state : int or None, default=None
        Seed controlling both the bootstrap sampling and each tree's
        feature subsampling, for reproducibility.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        random_state=None,
    ):
        super().__init__()

        if n_estimators <= 0:
            raise InvalidParameterError(
                "n_estimators must be a positive integer."
            )

        if max_depth is not None and max_depth <= 0:
            raise InvalidParameterError(
                "max_depth must be a positive integer or None."
            )

        if min_samples_split < 2:
            raise InvalidParameterError(
                "min_samples_split must be at least 2."
            )

        if criterion not in ("gini", "entropy"):
            raise InvalidParameterError(
                "criterion must be 'gini' or 'entropy'."
            )

        # Reuses the same validation logic as DecisionTreeClassifier
        # rather than duplicating the if/elif chain here.
        DecisionTreeClassifier._validate_max_features(max_features)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        """
        Train the forest: fit n_estimators Decision Trees, each on an
        independent bootstrap sample of (X, y).
        """

        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        rng = np.random.default_rng(self.random_state)

        self.trees_ = []

        for i in range(self.n_estimators):

            bootstrap_idx = rng.integers(0, n_samples, size=n_samples)
            X_sample = X[bootstrap_idx]
            y_sample = y[bootstrap_idx]

            # Each tree gets its own seed (derived from the forest's rng)
            # so their internal feature subsampling isn't all identical.
            tree_seed = int(rng.integers(0, 2**31 - 1))

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=tree_seed,
            )

            tree.fit(X_sample, y_sample)

            self.trees_.append(tree)

        self._mark_fitted()

        return self

    def predict(self, X):
        """
        Predict class labels by majority vote across all trees.
        """

        self._check_is_fitted(["trees_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        # Shape (n_estimators, n_samples): each row is one tree's
        # predictions for every sample.
        tree_predictions = np.array([tree.predict(X) for tree in self.trees_])

        predictions = np.array(
            [
                Counter(tree_predictions[:, i]).most_common(1)[0][0]
                for i in range(X.shape[0])
            ]
        )

        return predictions

    def score(self, X, y):
        """
        Return the classification accuracy of the forest on the given data.
        """

        y_pred = self.predict(X)

        return accuracy_score(np.asarray(y), y_pred)