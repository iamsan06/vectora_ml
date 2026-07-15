"""
Preprocessing utilities for Vectora-ML.
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import DimensionMismatchError
from vectora_ml.utils.validation import check_array


class StandardScaler(BaseEstimator):
    """
    Standardize features by removing the mean and scaling
    to unit variance.
    """

    def __init__(self):
        super().__init__()
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Compute the mean and standard deviation of each feature.
        """

        X = check_array(X)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Prevent division by zero
        self.std_[self.std_ == 0] = 1.0

        self.n_features_in_ = X.shape[1]

        self._mark_fitted()

        return self

    def transform(self, X):
        """
        Standardize the input data.
        """

        self._check_is_fitted(["mean_", "std_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )

        return (X - self.mean_) / self.std_

    def fit_transform(self, X, y=None):
        """
        Fit the scaler and transform the data.
        """

        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """
        Undo standardization.
        """

        self._check_is_fitted(["mean_", "std_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )

        return X * self.std_ + self.mean_


class MinMaxScaler(BaseEstimator):
    """
    Scale features into the range [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.min_ = None
        self.max_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Compute feature-wise minimum and maximum.
        """

        X = check_array(X)

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        self.n_features_in_ = X.shape[1]

        self._mark_fitted()

        return self

    def transform(self, X):
        """
        Scale features to the range [0, 1].
        """

        self._check_is_fitted(["min_", "max_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )

        denominator = self.max_ - self.min_
        denominator[denominator == 0] = 1.0

        return (X - self.min_) / denominator

    def fit_transform(self, X, y=None):
        """
        Fit the scaler and transform the data.
        """

        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """
        Undo Min-Max scaling.
        """

        self._check_is_fitted(["min_", "max_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, "
                f"but got {X.shape[1]}."
            )

        denominator = self.max_ - self.min_
        denominator[denominator == 0] = 1.0

        return X * denominator + self.min_


def train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=None,
):
    """
    Split arrays into random train and test subsets.
    """

    X = check_array(X)
    y = np.asarray(y)

    if y.ndim != 1:
        raise DimensionMismatchError(
            "Target vector y must be one-dimensional."
        )

    if len(X) != len(y):
        raise DimensionMismatchError(
            f"X contains {len(X)} samples but y contains {len(y)} samples."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    rng = np.random.default_rng(random_state)

    indices = np.arange(len(X))

    if shuffle:
        rng.shuffle(indices)

    split = int(len(X) * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )