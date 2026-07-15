"""
Tests for vectora_ml.ensemble.random_forest.RandomForestClassifier
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

from vectora_ml.ensemble.random_forest import RandomForestClassifier
from vectora_ml.tree.decision_tree import DecisionTreeClassifier
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
def iris_data():
    data = load_iris()
    return sk_train_test_split(
        data.data, data.target, test_size=0.2, random_state=0
    )


@pytest.fixture
def wine_data():
    data = load_wine()
    return sk_train_test_split(
        data.data, data.target, test_size=0.2, random_state=0
    )


# ---------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------

def test_invalid_n_estimators_raises():
    with pytest.raises(InvalidParameterError):
        RandomForestClassifier(n_estimators=0)


def test_invalid_max_depth_raises():
    with pytest.raises(InvalidParameterError):
        RandomForestClassifier(max_depth=0)


def test_invalid_criterion_raises():
    with pytest.raises(InvalidParameterError):
        RandomForestClassifier(criterion="bogus")


def test_invalid_max_features_raises():
    with pytest.raises(InvalidParameterError):
        RandomForestClassifier(max_features="bogus")


# ---------------------------------------------------------------------
# Fit / predict behavior
# ---------------------------------------------------------------------

def test_not_fitted_error_before_fit():
    model = RandomForestClassifier()
    X = np.array([[0.0, 0.0]])

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_fit_sets_expected_attributes(iris_data):
    X_train, X_test, y_train, y_test = iris_data

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_train, y_train)

    assert hasattr(model, "trees_")
    assert len(model.trees_) == 10
    assert all(isinstance(t, DecisionTreeClassifier) for t in model.trees_)
    assert model._is_fitted is True


def test_dimension_mismatch_on_predict(iris_data):
    X_train, X_test, y_train, y_test = iris_data

    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)

    with pytest.raises(DimensionMismatchError):
        model.predict(X_test[:, :2])


def test_trees_are_not_all_identical(iris_data):
    """
    Bootstrap sampling + per-split feature subsampling should mean the
    trees in the forest aren't all trained on the same data in the
    same way. This checks at least some diversity exists — if every
    tree were identical, the "forest" would just be n copies of one
    tree, defeating the whole point.
    """

    X_train, X_test, y_train, y_test = iris_data

    model = RandomForestClassifier(
        n_estimators=20, max_depth=3, max_features="sqrt", random_state=0
    )
    model.fit(X_train, y_train)

    predictions_per_tree = [tuple(t.predict(X_test)) for t in model.trees_]
    unique_prediction_sets = set(predictions_per_tree)

    assert len(unique_prediction_sets) > 1


def test_default_max_features_is_sqrt():
    """
    max_features="sqrt" is what makes this a Random Forest rather than
    plain bagging — worth locking in as the default.
    """

    model = RandomForestClassifier()
    assert model.max_features == "sqrt"


# ---------------------------------------------------------------------
# Correctness vs sklearn
# ---------------------------------------------------------------------

def test_accuracy_matches_sklearn_on_iris(iris_data):
    X_train, X_test, y_train, y_test = iris_data

    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
    model.fit(X_train, y_train)
    our_acc = model.score(X_test, y_test)

    sklearn_model = SklearnRandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=0
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_acc = accuracy_score(y_test, sklearn_model.predict(X_test))

    assert abs(our_acc - sklearn_acc) < 0.05


def test_accuracy_matches_sklearn_on_wine(wine_data):
    X_train, X_test, y_train, y_test = wine_data

    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
    model.fit(X_train, y_train)
    our_acc = model.score(X_test, y_test)

    sklearn_model = SklearnRandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=0
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_acc = accuracy_score(y_test, sklearn_model.predict(X_test))

    # Wine's test split is only 36 samples, so a single misclassified
    # sample swings accuracy by ~2.8% — a slightly looser tolerance
    # than Iris here avoids flakiness without hiding a real regression.
    assert abs(our_acc - sklearn_acc) < 0.1


def test_forest_at_least_as_good_as_single_tree(iris_data):
    """
    Not a strict guarantee in general, but on a clean, simple dataset
    like Iris a forest should be at least competitive with — not
    dramatically worse than — a single well-tuned tree.
    """

    X_train, X_test, y_train, y_test = iris_data

    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X_train, y_train)

    forest = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0)
    forest.fit(X_train, y_train)

    assert forest.score(X_test, y_test) >= tree.score(X_test, y_test) - 0.05