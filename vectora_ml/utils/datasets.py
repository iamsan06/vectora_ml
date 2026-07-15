"""
Dataset generation utilities.

These functions generate synthetic datasets for testing
and experimenting with machine learning algorithms.
"""

import numpy as np

def make_regression(
    n_samples=100,
    n_features=1,
    noise=0.0,
    bias=0.0,
    random_state=None,
):
    """
    Generate a synthetic regression dataset.

    Returns
    -------
    X : ndarray
    y : ndarray
    coef : ndarray
    """

    rng = np.random.default_rng(random_state)

    X = rng.normal(size=(n_samples, n_features))

    coef = rng.normal(size=n_features)

    y = X @ coef + bias

    if noise > 0:
        y += rng.normal(scale=noise, size=n_samples)

    return X, y, coef


def make_classification(
    n_samples=100,
    n_features=2,
    random_state=None,
):
    """
    Generate a simple binary classification dataset.
    """

    rng = np.random.default_rng(random_state)

    X = rng.normal(size=(n_samples, n_features))

    weights = rng.normal(size=n_features)

    scores = X @ weights

    y = (scores > 0).astype(int)

    return X, y


def make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.0,
    random_state=None,
):
    """
    Generate clustered data for clustering algorithms.
    """

    rng = np.random.default_rng(random_state)

    samples_per_cluster = n_samples // centers

    X = []
    y = []

    cluster_centers = rng.uniform(
        -10,
        10,
        size=(centers, n_features),
    )

    for i in range(centers):

        points = rng.normal(
            loc=cluster_centers[i],
            scale=cluster_std,
            size=(samples_per_cluster, n_features),
        )

        X.append(points)

        y.extend([i] * samples_per_cluster)

    X = np.vstack(X)

    y = np.array(y)

    return X, y