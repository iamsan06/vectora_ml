"""
Principal Component Analysis via eigendecomposition of the covariance matrix.
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)
from vectora_ml.utils.validation import check_array


class PCA(BaseEstimator):
    """
    Principal Component Analysis.

    Finds the directions (principal components) along which the data
    varies the most, by eigendecomposing the feature covariance
    matrix. The eigenvectors are the directions; their eigenvalues are
    how much variance lies along each one. Keeping only the top
    `n_components` (by eigenvalue) and projecting the data onto them
    is dimensionality reduction: the directions that carry the least
    information are the ones dropped.

    Unlike every other estimator in this library, PCA is unsupervised
    — fit() takes no y, and there's no predict(). Use transform() to
    project data into the reduced space.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to keep. None keeps all of
        them (a full change of basis, no dimensionality actually
        reduced — mostly useful for inspecting explained_variance_
        across every component before choosing how many to keep).
    """

    def __init__(self, n_components=None):
        super().__init__()

        if n_components is not None and n_components <= 0:
            raise InvalidParameterError(
                "n_components must be a positive integer or None."
            )

        self.n_components = n_components

    def fit(self, X, y=None):
        """
        Compute the principal components: center the data, build the
        covariance matrix, eigendecompose it, and keep the top
        `n_components` eigenvectors by eigenvalue.
        """

        X = check_array(X)

        n_samples, n_features = X.shape

        n_components = (
            self.n_components if self.n_components is not None else n_features
        )

        if n_components > n_features:
            raise InvalidParameterError(
                f"n_components ({n_components}) cannot exceed the number "
                f"of features ({n_features})."
            )

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Feature-by-feature covariance matrix. Using n_samples - 1
        # (Bessel's correction) for an unbiased estimate, same
        # convention as np.cov's default.
        covariance = (X_centered.T @ X_centered) / (n_samples - 1)

        # eigh (not eig) because covariance is guaranteed symmetric —
        # eigh is faster and, unlike eig, guarantees real eigenvalues
        # returned in a numerically stable way. eigh returns them in
        # ASCENDING order, so they're reversed below.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # components_ is (n_components, n_features): each ROW is one
        # principal component direction, which is what makes the
        # projection in transform() a clean X_centered @ components_.T.
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]

        total_variance = np.sum(eigenvalues)

        if total_variance > 0:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )
        else:
            self.explained_variance_ratio_ = np.zeros(n_components)

        self.n_features_in_ = n_features

        self._mark_fitted()

        return self

    def transform(self, X):
        """
        Project X onto the principal components found during fit().
        """

        self._check_is_fitted(["components_", "mean_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def fit_transform(self, X, y=None):
        """
        Fit PCA and transform X in one step.
        """

        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct data back into the original feature space from its
        projection. Lossy unless n_components == n_features_in_, since
        the components that were dropped are gone — this reconstructs
        the best approximation using only the components that were kept.
        """

        self._check_is_fitted(["components_", "mean_", "n_features_in_"])

        X_transformed = np.asarray(X_transformed, dtype=float)

        return X_transformed @ self.components_ + self.mean_