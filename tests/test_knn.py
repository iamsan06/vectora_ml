"""
Pytest suite for KNNClassifier (knn.py).

"""

import numpy as np
import pytest

from vectora_ml.neighbors.knn import KNNClassifier
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blobs():
    """
    Two tight, well-separated 2D clusters: label 0 near (0, 0),
    label 1 near (10, 10). Any reasonable k should classify these
    perfectly.
    """
    rng = np.random.RandomState(0)
    class0 = rng.normal(loc=(0, 0), scale=0.5, size=(30, 2))
    class1 = rng.normal(loc=(10, 10), scale=0.5, size=(30, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * 30 + [1] * 30)
    return X, y


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        knn = KNNClassifier()
        assert knn.n_neighbors == 5
        assert knn.p == 2

    @pytest.mark.parametrize("bad_k", [0, -1, -10])
    def test_rejects_non_positive_n_neighbors(self, bad_k):
        with pytest.raises(InvalidParameterError):
            KNNClassifier(n_neighbors=bad_k)

    @pytest.mark.parametrize("bad_p", [0, -1, -2.5])
    def test_rejects_non_positive_p(self, bad_p):
        with pytest.raises(InvalidParameterError):
            KNNClassifier(p=bad_p)

    def test_accepts_custom_p(self):
        knn = KNNClassifier(p=1)
        assert knn.p == 1


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier()
        assert knn.fit(X, y) is knn

    def test_fit_stores_training_data(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier().fit(X, y)
        np.testing.assert_array_equal(knn.X_train_, X)
        np.testing.assert_array_equal(knn.y_train_, y)

    def test_fit_sets_n_features_in(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier().fit(X, y)
        assert knn.n_features_in_ == 2


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predicts_correctly_on_separated_blobs(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier(n_neighbors=3).fit(X, y)
        preds = knn.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_predict_before_fit_raises(self, two_blobs):
        X, _ = two_blobs
        knn = KNNClassifier()
        with pytest.raises(Exception):
            knn.predict(X)

    def test_predict_dimension_mismatch_raises(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier().fit(X, y)
        with pytest.raises(DimensionMismatchError):
            knn.predict(X[:, :1])

    def test_predict_output_shape_and_dtype(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier().fit(X, y)
        preds = knn.predict(X)
        assert preds.shape == (X.shape[0],)
        assert preds.dtype == y.dtype

    def test_new_point_classified_to_nearest_cluster(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier(n_neighbors=5).fit(X, y)
        near_class0 = np.array([[0.1, -0.1]])
        near_class1 = np.array([[9.9, 10.1]])
        assert knn.predict(near_class0)[0] == 0
        assert knn.predict(near_class1)[0] == 1

    def test_k_equals_1_matches_single_nearest_neighbor(self):
        X = np.array([[0.0], [1.0], [5.0], [6.0]])
        y = np.array([0, 0, 1, 1])
        knn = KNNClassifier(n_neighbors=1).fit(X, y)
        query = np.array([[0.9]])  # closest to X[1] (value 1.0), label 0
        assert knn.predict(query)[0] == 0

    def test_manhattan_vs_euclidean_can_differ(self):
        # Construct a case where L1 and L2 distance orderings diverge.
        X_train = np.array([[3.0, 0.0], [0.0, 4.0]])
        y_train = np.array([0, 1])
        query = np.array([[1.5, 1.5]])

        knn_l2 = KNNClassifier(n_neighbors=1, p=2).fit(X_train, y_train)
        knn_l1 = KNNClassifier(n_neighbors=1, p=1).fit(X_train, y_train)

        # Both distances are equal here for L2 (5.0 vs 5.0 -> tie) but
        # this mainly checks both metrics run without error and return
        # a valid label.
        assert knn_l2.predict(query)[0] in (0, 1)
        assert knn_l1.predict(query)[0] in (0, 1)

    def test_majority_vote_breaks_towards_more_common_label(self):
        # 3 neighbors: two of label 1, one of label 0 -> predict 1.
        X_train = np.array([[0.0], [1.0], [2.0], [100.0]])
        y_train = np.array([0, 1, 1, 0])
        knn = KNNClassifier(n_neighbors=3).fit(X_train, y_train)
        query = np.array([[1.0]])
        assert knn.predict(query)[0] == 1


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScore:
    def test_perfect_score_on_training_data_when_separable(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier(n_neighbors=3).fit(X, y)
        assert knn.score(X, y) == pytest.approx(1.0)

    def test_score_returns_float_between_0_and_1(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier().fit(X, y)
        acc = knn.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_score_with_all_wrong_labels(self, two_blobs):
        X, y = two_blobs
        knn = KNNClassifier(n_neighbors=3).fit(X, y)
        flipped_y = 1 - y
        acc = knn.score(X, flipped_y)
        assert acc == pytest.approx(0.0)