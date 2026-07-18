"""
K-Nearest Neighbors Classifier.
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)
from vectora_ml.core.metrics import accuracy_score
from vectora_ml.utils.validation import check_array, check_X_y


class KNNClassifier(BaseEstimator):
    """
    K-Nearest Neighbors Classifier.

    A "lazy learner" — fit() just stores the training data; there's no
    actual model to train. All the real work happens at predict() time:
    for each query point, find the k closest training points and let
    them vote on the label.

    Distance is Minkowski distance of order `p`:
        d(x, z) = (sum(|x_i - z_i| ** p)) ** (1/p)
    p=2 is standard Euclidean distance (the default); p=1 is Manhattan
    distance. Both are special cases of the same family.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors (k) to vote on each prediction.

    p : int or float, default=2
        Order of the Minkowski distance. 2 = Euclidean, 1 = Manhattan.
    """

    def __init__(self, n_neighbors=5, p=2):
        super().__init__()

        if n_neighbors <= 0:
            raise InvalidParameterError(
                "n_neighbors must be a positive integer."
            )

        if p <= 0:
            raise InvalidParameterError("p must be positive.")

        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X, y):
        """
        Store the training data. No actual computation happens here —
        that's what makes KNN a "lazy" learner, as opposed to
        LinearRegression or DecisionTreeClassifier which do all their
        work upfront in fit().
        """

        X, y = check_X_y(X, y)

        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]

        self._mark_fitted()

        return self

    def _distances(self, X):
        """
        Pairwise Minkowski distance between every row in X (queries)
        and every row in X_train_ (stored training points). Returns an
        array of shape (n_queries, n_train_samples).

        Broadcasting X[:, None, :] - X_train_[None, :, :] gives every
        (query, train_point) difference in one shot, at the cost of an
        (n_queries, n_train, n_features) intermediate array — fine for
        the dataset sizes this library targets, but the reason real
        KNN implementations use k-d trees instead of brute force.
        """

        diffs = X[:, None, :] - self.X_train_[None, :, :]

        return np.sum(np.abs(diffs) ** self.p, axis=2) ** (1 / self.p)

    def predict(self, X):
        """
        Predict class labels by majority vote among each query point's
        k nearest training neighbors.
        """

        self._check_is_fitted(["X_train_", "y_train_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        distances = self._distances(X)

        # Indices of the k closest training points, per query row.
        neighbor_indices = np.argsort(distances, axis=1)[:, : self.n_neighbors]

        predictions = np.empty(X.shape[0], dtype=self.y_train_.dtype)

        for i, neighbors in enumerate(neighbor_indices):
            neighbor_labels = self.y_train_[neighbors]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            predictions[i] = values[np.argmax(counts)]

        return predictions

    def score(self, X, y):
        """
        Return the classification accuracy of the model on the given data.
        """

        y_pred = self.predict(X)

        return accuracy_score(np.asarray(y), y_pred)