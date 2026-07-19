"""
Pytest suite for SVC (svm.py).

"""

import numpy as np
import pytest

from vectora_ml.svm.svm import SVC
from vectora_ml.core.exceptions import (
    ConvergenceError,
    DimensionMismatchError,
    InvalidParameterError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separable_data():
    """
    Two well-separated 2D clusters, linearly separable by a wide
    margin: label 0 near (-3, -3), label 1 near (3, 3).
    """
    rng = np.random.RandomState(0)
    class0 = rng.normal(loc=(-3, -3), scale=0.5, size=(50, 2))
    class1 = rng.normal(loc=(3, 3), scale=0.5, size=(50, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        svc = SVC()
        assert svc.alpha == 0.01
        assert svc.learning_rate == 0.001
        assert svc.max_iter == 1000
        assert svc.tol == 1e-6
        assert svc.loss_history_ == []

    def test_rejects_negative_alpha(self):
        with pytest.raises(InvalidParameterError):
            SVC(alpha=-0.1)

    def test_allows_zero_alpha(self):
        # alpha must be non-negative, so 0 should be fine.
        SVC(alpha=0)

    @pytest.mark.parametrize("bad_lr", [0, -0.01])
    def test_rejects_non_positive_learning_rate(self, bad_lr):
        with pytest.raises(InvalidParameterError):
            SVC(learning_rate=bad_lr)

    @pytest.mark.parametrize("bad_iter", [0, -5])
    def test_rejects_non_positive_max_iter(self, bad_iter):
        with pytest.raises(InvalidParameterError):
            SVC(max_iter=bad_iter)

    @pytest.mark.parametrize("bad_tol", [0, -1e-6])
    def test_rejects_non_positive_tol(self, bad_tol):
        with pytest.raises(InvalidParameterError):
            SVC(tol=bad_tol)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=200)
        assert svc.fit(X, y) is svc

    def test_fit_sets_expected_attributes(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=200).fit(X, y)
        assert svc.coef_.shape == (2,)
        assert isinstance(svc.intercept_, float)
        assert svc.n_features_in_ == 2
        assert svc.n_iter_ == len(svc.loss_history_)
        assert svc.final_loss_ == svc.loss_history_[-1]

    def test_rejects_labels_outside_zero_one(self, separable_data):
        X, y = separable_data
        bad_y = y.copy()
        bad_y[0] = 2
        svc = SVC(max_iter=50)
        with pytest.raises(InvalidParameterError):
            svc.fit(X, bad_y)

    def test_loss_decreases_over_training(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=300, learning_rate=0.01).fit(X, y)
        history = svc.loss_history_
        # Final loss should be substantially lower than the first
        # recorded loss for an easily-separable dataset.
        assert history[-1] < history[0]

    def test_converges_and_separates_well(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=2000, learning_rate=0.05, alpha=0.001).fit(X, y)
        preds = svc.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.95

    def test_early_stopping_via_tol(self, separable_data):
        X, y = separable_data
        # A very loose tol should trigger the early-stopping branch
        # well before max_iter is reached.
        svc = SVC(max_iter=5000, tol=10.0).fit(X, y)
        assert svc.n_iter_ < 5000

    def test_diverges_raises_convergence_error(self, separable_data):
        X, y = separable_data
        # An absurdly large learning rate should blow up the loss into
        # NaN/Inf territory quickly.
        svc = SVC(max_iter=1000, learning_rate=1e10)
        with pytest.raises(ConvergenceError):
            svc.fit(X, y)

    def test_refit_resets_loss_history(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=50).fit(X, y)
        first_len = len(svc.loss_history_)
        svc.fit(X, y)
        # Refitting on the same data with the same settings should
        # produce a loss history of the same length, not an
        # accumulation across both fits.
        assert len(svc.loss_history_) == first_len


# ---------------------------------------------------------------------------
# decision_function()
# ---------------------------------------------------------------------------

class TestDecisionFunction:
    def test_decision_function_before_fit_raises(self, separable_data):
        X, _ = separable_data
        svc = SVC()
        with pytest.raises(Exception):
            svc.decision_function(X)

    def test_decision_function_dimension_mismatch_raises(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=100).fit(X, y)
        with pytest.raises(DimensionMismatchError):
            svc.decision_function(X[:, :1])

    def test_decision_function_sign_matches_class(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=1000, learning_rate=0.05).fit(X, y)
        scores = svc.decision_function(X)
        preds = (scores >= 0).astype(int)
        np.testing.assert_array_equal(preds, svc.predict(X))


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_output_is_binary(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=500).fit(X, y)
        preds = svc.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_new_points(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=1000, learning_rate=0.05, alpha=0.001).fit(X, y)
        near_class0 = np.array([[-3.0, -3.0]])
        near_class1 = np.array([[3.0, 3.0]])
        assert svc.predict(near_class0)[0] == 0
        assert svc.predict(near_class1)[0] == 1


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_high_on_separable_data(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=2000, learning_rate=0.05, alpha=0.001).fit(X, y)
        assert svc.score(X, y) > 0.95

    def test_score_between_zero_and_one(self, separable_data):
        X, y = separable_data
        svc = SVC(max_iter=100).fit(X, y)
        acc = svc.score(X, y)
        assert 0.0 <= acc <= 1.0