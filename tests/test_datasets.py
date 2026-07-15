import numpy as np

from vectora_ml.utils.datasets import (
    make_regression,
    make_classification,
    make_blobs,
)


def test_make_regression():

    X, y, coef = make_regression(
        n_samples=100,
        n_features=5,
        random_state=42,
    )

    assert X.shape == (100, 5)
    assert y.shape == (100,)
    assert coef.shape == (5,)


def test_make_classification():

    X, y = make_classification(
        n_samples=200,
        n_features=4,
        random_state=42,
    )

    assert X.shape == (200, 4)
    assert y.shape == (200,)

    assert set(np.unique(y)) == {0, 1}


def test_make_blobs():

    X, y = make_blobs(
        n_samples=300,
        centers=3,
        random_state=42,
    )

    assert X.shape[1] == 2

    assert len(np.unique(y)) == 3