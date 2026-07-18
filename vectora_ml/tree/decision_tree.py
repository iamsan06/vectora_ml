"""
Decision Tree Classifier via recursive binary splitting.
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)
from vectora_ml.core.metrics import accuracy_score
from vectora_ml.utils.validation import check_array, check_X_y
from vectora_ml.tree.node import Node


class DecisionTreeClassifier(BaseEstimator):
    """
    Decision Tree Classifier trained via recursive binary splitting.

    At each node, every (feature, threshold) pair is evaluated and the
    split that most reduces impurity is kept. Splitting continues until
    a stopping condition is hit, at which point the node becomes a leaf
    that predicts the majority class of the samples that reached it.

    Supports per-sample weighting via `fit(X, y, sample_weight=...)`.
    With uniform weights (the default) this behaves exactly like a
    plain decision tree; non-uniform weights let boosting algorithms
    like AdaBoost reuse this class as their weak learner.

    After fitting, `feature_importances_` reports how much each
    feature contributed to reducing impurity across the whole tree.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. None means nodes are expanded until
        all leaves are pure or another stopping condition is hit.

    min_samples_split : int, default=2
        Minimum number of samples a node must have to be split further.

    criterion : {"gini", "entropy"}, default="gini"
        Impurity measure used to evaluate splits.

    max_features : None, "sqrt", "log2", int, or float, default=None
        Number of features to consider at each split. Restricting this
        decorrelates trees from each other, which is what
        RandomForestClassifier relies on this class for.

    random_state : int or None, default=None
        Seed controlling feature subsampling when max_features is set.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None,
    ):
        super().__init__()

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

        self._validate_max_features(max_features)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

    @staticmethod
    def _validate_max_features(max_features):
        """
        max_features accepts several shapes (None / str / int / float),
        so validation is pulled out into its own method rather than
        cluttering __init__ with a long if/elif chain. Also reused as-is
        by RandomForestClassifier so the two never drift apart.
        """

        if max_features is None:
            return

        if isinstance(max_features, str):
            if max_features not in ("sqrt", "log2"):
                raise InvalidParameterError(
                    "max_features string must be 'sqrt' or 'log2'."
                )
            return

        # bool is a subclass of int in Python, so this must be checked
        # before the int check below, or True/False would silently pass.
        if isinstance(max_features, bool):
            raise InvalidParameterError(
                "max_features must be None, 'sqrt', 'log2', an int, or a float."
            )

        if isinstance(max_features, int):
            if max_features <= 0:
                raise InvalidParameterError("max_features must be positive.")
            return

        if isinstance(max_features, float):
            if not 0 < max_features <= 1:
                raise InvalidParameterError(
                    "max_features as a float must be in (0, 1]."
                )
            return

        raise InvalidParameterError(
            "max_features must be None, 'sqrt', 'log2', an int, or a float."
        )

    # ------------------------------------------------------------------
    # Impurity measures (weighted — uniform weights reduce these to the
    # standard unweighted formulas)
    # ------------------------------------------------------------------

    @staticmethod
    def _gini(y, weights):
        """
        Weighted Gini impurity: probability that two samples drawn
        (proportional to weight) from this node have different classes.
        0 = pure node, higher = more mixed.
        """

        total_weight = np.sum(weights)
        gini = 1.0

        for label in np.unique(y):
            p = np.sum(weights[y == label]) / total_weight
            gini -= p ** 2

        return gini

    @staticmethod
    def _entropy(y, weights):
        """
        Weighted Shannon entropy, in bits. Same intuition as Gini
        (0 = pure, higher = more mixed) via a different formula.
        """

        total_weight = np.sum(weights)
        entropy = 0.0

        for label in np.unique(y):
            p = np.sum(weights[y == label]) / total_weight

            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def _impurity(self, y, weights):
        if self.criterion == "gini":
            return self._gini(y, weights)

        return self._entropy(y, weights)

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _max_features_count(self, n_features):
        if self.max_features is None:
            return n_features

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))

            return max(1, int(np.log2(n_features)))  # "log2"

        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))

        return min(self.max_features, n_features)  # int

    def _select_feature_subset(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        count = self._max_features_count(n_features)

        return self._rng.choice(n_features, size=count, replace=False)

    def _best_split(self, X, y, weights, feature_indices):
        """
        Search the given features for the (feature, threshold) pair
        that most reduces weighted impurity, evaluating every midpoint
        between consecutive sorted unique values as a candidate
        threshold. Returns None if no split improves on the parent,
        otherwise (feature_index, threshold, gain).
        """

        total_weight = np.sum(weights)
        parent_impurity = self._impurity(y, weights)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_index in feature_indices:

            values = np.unique(X[:, feature_index])

            if len(values) < 2:
                continue

            candidate_thresholds = (values[:-1] + values[1:]) / 2

            for threshold in candidate_thresholds:

                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left, w_left = y[left_mask], weights[left_mask]
                y_right, w_right = y[right_mask], weights[right_mask]

                weighted_impurity = (
                    (np.sum(w_left) / total_weight) * self._impurity(y_left, w_left)
                    + (np.sum(w_right) / total_weight) * self._impurity(y_right, w_right)
                )

                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        if best_feature is None:
            return None

        return best_feature, best_threshold, best_gain

    @staticmethod
    def _majority_class(y, weights):
        labels = np.unique(y)
        weighted_counts = [np.sum(weights[y == label]) for label in labels]

        return labels[np.argmax(weighted_counts)]

    def _build_tree(self, X, y, weights, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or n_labels == 1
        ):
            return Node(value=self._majority_class(y, weights))

        feature_indices = self._select_feature_subset(n_features)

        split = self._best_split(X, y, weights, feature_indices)

        if split is None:
            return Node(value=self._majority_class(y, weights))

        feature_index, threshold, gain = split

        # Feature importance: how much this single split reduced
        # impurity, weighted by what fraction of the TOTAL training
        # weight passed through this node. A split near the root (most
        # samples affected) counts for more than one deep in the tree
        # (few samples affected) — same idea sklearn uses.
        node_weight = np.sum(weights)
        self._raw_importances[feature_index] += (
            (node_weight / self._root_weight_total) * gain
        )

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(
            X[left_mask], y[left_mask], weights[left_mask], depth + 1
        )
        right = self._build_tree(
            X[right_mask], y[right_mask], weights[right_mask], depth + 1
        )

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left,
            right=right,
        )

    def fit(self, X, y, sample_weight=None):
        """
        Train the decision tree by recursively splitting on the
        feature/threshold pair that most reduces weighted impurity at
        each node. Also computes `feature_importances_`.

        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights. Defaults to uniform weights.
        """

        X, y = check_X_y(X, y)

        if sample_weight is None:
            weights = np.ones(len(y))
        else:
            weights = np.asarray(sample_weight, dtype=float)

            if weights.shape[0] != len(y):
                raise DimensionMismatchError(
                    f"sample_weight has {weights.shape[0]} entries but "
                    f"y has {len(y)}."
                )

        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)

        self._root_weight_total = np.sum(weights)
        self._raw_importances = np.zeros(self.n_features_in_)

        self.root_ = self._build_tree(X, y, weights, depth=0)

        # Normalize so importances sum to 1 (0 if the tree is a single
        # leaf and never split at all).
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
        Predict class labels for samples in X by routing each sample
        from the root to a leaf.
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
        Return the classification accuracy of the model on the given data.
        """

        y_pred = self.predict(X)

        return accuracy_score(np.asarray(y), y_pred)

    def print_tree(self, feature_names=None):
        """
        Print an indented text representation of the tree.
        """

        self._check_is_fitted(["root_"])

        self._print_node(self.root_, feature_names, depth=0)

    def _print_node(self, node, feature_names, depth):
        indent = "  " * depth

        if node.is_leaf():
            print(f"{indent}Predict -> {node.value}")
            return

        if feature_names is not None:
            name = feature_names[node.feature_index]
        else:
            name = f"X[{node.feature_index}]"

        print(f"{indent}if {name} <= {node.threshold:.3f}:")
        self._print_node(node.left, feature_names, depth + 1)

        print(f"{indent}else:")
        self._print_node(node.right, feature_names, depth + 1)