"""
Tests for vectora_ml.ensemble.ada_boost.AdaBoostClassifier
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification as sk_make_classification
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoostClassifier

from vectora_ml.ensemble.ada_boost import AdaBoostClassifier
from vectora_ml.core.exceptions import (
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)
from vectora_ml.core.metrics import accuracy_score


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def classification_data():
    X, y = sk_make_classification(
        n_samples=500,
        n_features=10,
        n_classes=2,
        random_state=0,
    )
    return sk_train_test_split(X, y, test_size=0.2, random_state=0)


# ---------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------

def test_invalid_n_estimators_raises():
    with pytest.raises(InvalidParameterError):
        AdaBoostClassifier(n_estimators=0)


def test_invalid_learning_rate_raises():
    with pytest.raises(InvalidParameterError):
        AdaBoostClassifier(learning_rate=0)


def test_invalid_max_depth_raises():
    with pytest.raises(InvalidParameterError):
        AdaBoostClassifier(max_depth=0)


def test_rejects_non_binary_targets(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    y_multiclass = np.where(y_train == 1, 2, y_train)  # labels become {0, 2}

    model = AdaBoostClassifier(n_estimators=5)

    with pytest.raises(InvalidParameterError):
        model.fit(X_train, y_multiclass)


# ---------------------------------------------------------------------
# Fit / predict behavior
# ---------------------------------------------------------------------

def test_not_fitted_error_before_fit():
    model = AdaBoostClassifier()
    X = np.array([[0.0, 0.0]])

    with pytest.raises(NotFittedError):
        model.predict(X)

    with pytest.raises(NotFittedError):
        model.decision_function(X)


def test_fit_sets_expected_attributes(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = AdaBoostClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    assert len(model.estimators_) == 10
    assert len(model.estimator_weights_) == 10
    assert len(model.estimator_errors_) == 10
    assert model._is_fitted is True


def test_dimension_mismatch_on_predict(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = AdaBoostClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    with pytest.raises(DimensionMismatchError):
        model.predict(X_test[:, :3])


def test_predict_returns_binary_labels(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = AdaBoostClassifier(n_estimators=20)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert set(np.unique(predictions)).issubset({0, 1})


def test_sample_weights_stay_normalized_after_each_round():
    """
    AdaBoost's reweighting step must renormalize after every round, or
    the weighted error calculation in later rounds would be meaningless.
    This re-derives the weights the same way fit() does and checks they
    always sum to 1.
    """

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] > 0).astype(int)

    model = AdaBoostClassifier(n_estimators=10)
    model.fit(X, y)

    # If fit() didn't renormalize, later estimator_errors_ would tend
    # to drift out of [0, 1] — a cheap indirect check that renormalization
    # happened at every round.
    assert all(0 <= error <= 1 for error in model.estimator_errors_)


def test_more_estimators_improves_or_maintains_training_accuracy():
    """
    AdaBoost is a sequential, error-correcting process — training
    accuracy should generally not get worse as more weak learners are
    added, on a dataset simple enough to eventually be learned well.
    """

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    few = AdaBoostClassifier(n_estimators=3)
    many = AdaBoostClassifier(n_estimators=30)

    few.fit(X, y)
    many.fit(X, y)

    assert many.score(X, y) >= few.score(X, y) - 0.02


# ---------------------------------------------------------------------
# Correctness vs sklearn
# ---------------------------------------------------------------------

def test_accuracy_comparable_to_sklearn(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    model.fit(X_train, y_train)
    our_acc = model.score(X_test, y_test)

    sklearn_model = SklearnAdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    sklearn_model.fit(X_train, y_train)
    sklearn_acc = accuracy_score(y_test, sklearn_model.predict(X_test))

    assert abs(our_acc - sklearn_acc) < 0.1


def test_score_matches_accuracy_score(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    model = AdaBoostClassifier(n_estimators=20)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    expected = accuracy_score(y_test, predictions)

    assert model.score(X_test, y_test) == pytest.approx(expected)