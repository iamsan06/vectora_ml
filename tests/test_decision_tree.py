"""
Tests for vectora_ml.tree.decision_tree.DecisionTreeClassifier
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

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

def test_invalid_max_depth_raises():
    with pytest.raises(InvalidParameterError):
        DecisionTreeClassifier(max_depth=0)

    with pytest.raises(InvalidParameterError):
        DecisionTreeClassifier(max_depth=-1)


def test_invalid_min_samples_split_raises():
    with pytest.raises(InvalidParameterError):
        DecisionTreeClassifier(min_samples_split=1)


def test_invalid_criterion_raises():
    with pytest.raises(InvalidParameterError):
        DecisionTreeClassifier(criterion="chi_square")


def test_invalid_max_features_string_raises():
    with pytest.raises(InvalidParameterError):
        DecisionTreeClassifier(max_features="bogus")


def test_invalid_max_features_bool_raises():
    # bool is a subclass of int in Python — must not silently pass as int.
    with pytest.raises(InvalidParameterError):
        DecisionTreeClassifier(max_features=True)


def test_invalid_max_features_float_raises():
    with pytest.raises(InvalidParameterError):
        DecisionTreeClassifier(max_features=1.5)

    with pytest.raises(InvalidParameterError):
        DecisionTreeClassifier(max_features=0.0)


def test_valid_max_features_values_accepted():
    for value in [None, "sqrt", "log2", 2, 0.5]:
        model = DecisionTreeClassifier(max_features=value)
        assert model.max_features == value


# ---------------------------------------------------------------------
# Fit / predict behavior
# ---------------------------------------------------------------------

def test_not_fitted_error_before_fit():
    model = DecisionTreeClassifier()
    X = np.array([[0.0, 0.0]])

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_fit_sets_expected_attributes(iris_data):
    X_train, X_test, y_train, y_test = iris_data

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    assert hasattr(model, "root_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == X_train.shape[1]
    assert model._is_fitted is True


def test_dimension_mismatch_on_predict(iris_data):
    X_train, X_test, y_train, y_test = iris_data

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    bad_X = X_test[:, :2]

    with pytest.raises(DimensionMismatchError):
        model.predict(bad_X)


def test_perfectly_separable_data_reaches_zero_training_error():
    """
    An unconstrained tree (no max_depth) should be able to perfectly
    memorize a small, cleanly separable dataset.
    """

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier()
    model.fit(X, y)

    predictions = model.predict(X)

    assert np.array_equal(predictions, y)


def test_max_depth_limits_tree_growth():
    """
    A depth-1 tree (a single stump) should generally do worse than an
    unconstrained tree on data that isn't linearly separable by a
    single split — this exercises max_depth actually being respected.
    """

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    # XOR-like pattern: not separable by a single axis-aligned split.
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    stump = DecisionTreeClassifier(max_depth=1)
    stump.fit(X, y)

    full_tree = DecisionTreeClassifier(max_depth=10)
    full_tree.fit(X, y)

    stump_acc = stump.score(X, y)
    full_acc = full_tree.score(X, y)

    assert full_acc > stump_acc


def test_gini_and_entropy_produce_reasonable_accuracy(iris_data):
    X_train, X_test, y_train, y_test = iris_data

    gini_model = DecisionTreeClassifier(max_depth=3, criterion="gini")
    entropy_model = DecisionTreeClassifier(max_depth=3, criterion="entropy")

    gini_model.fit(X_train, y_train)
    entropy_model.fit(X_train, y_train)

    # Both are legitimate impurity measures and should land in the
    # same ballpark on a simple, well-separated dataset like Iris.
    assert gini_model.score(X_test, y_test) > 0.85
    assert entropy_model.score(X_test, y_test) > 0.85


# ---------------------------------------------------------------------
# sample_weight support (needed by AdaBoost)
# ---------------------------------------------------------------------

def test_fit_accepts_sample_weight():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    model = DecisionTreeClassifier()
    model.fit(X, y, sample_weight=weights)  # should not raise

    assert model._is_fitted is True


def test_sample_weight_mismatched_length_raises():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])
    weights = np.array([1.0, 1.0])  # wrong length

    model = DecisionTreeClassifier()

    with pytest.raises(DimensionMismatchError):
        model.fit(X, y, sample_weight=weights)


def test_uniform_sample_weight_matches_unweighted_fit():
    """
    Explicit uniform weights should produce the exact same tree as
    fitting with no weights at all — this is the property AdaBoost's
    first round (uniform weights) relies on implicitly.
    """

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] > 0).astype(int)

    unweighted = DecisionTreeClassifier(max_depth=3, random_state=0)
    unweighted.fit(X, y)

    weighted = DecisionTreeClassifier(max_depth=3, random_state=0)
    weighted.fit(X, y, sample_weight=np.ones(len(y)))

    np.testing.assert_array_equal(
        unweighted.predict(X), weighted.predict(X)
    )


def test_heavily_upweighted_samples_dominate_the_split():
    """
    If a small group of samples is given overwhelming weight, the tree
    should prioritize classifying THEM correctly, even at the cost of
    getting unweighted-majority samples wrong.
    """

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 0])  # last sample is the "odd one out"

    # Give the last sample enormous weight relative to the rest.
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])

    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X, y, sample_weight=weights)

    predictions = model.predict(X)

    # The heavily-weighted sample must be classified correctly.
    assert predictions[-1] == y[-1]


# ---------------------------------------------------------------------
# print_tree (should not raise)
# ---------------------------------------------------------------------

def test_print_tree_runs_without_error(iris_data, capsys):
    X_train, X_test, y_train, y_test = iris_data

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    model.print_tree()

    captured = capsys.readouterr()
    assert "Predict" in captured.out


# ---------------------------------------------------------------------
# Correctness vs sklearn
# ---------------------------------------------------------------------

def test_accuracy_matches_sklearn_on_iris(iris_data):
    X_train, X_test, y_train, y_test = iris_data

    model = DecisionTreeClassifier(max_depth=5, random_state=0)
    model.fit(X_train, y_train)
    our_acc = model.score(X_test, y_test)

    sklearn_model = SklearnDecisionTreeClassifier(max_depth=5, random_state=0)
    sklearn_model.fit(X_train, y_train)
    sklearn_acc = accuracy_score(y_test, sklearn_model.predict(X_test))

    assert abs(our_acc - sklearn_acc) < 0.05


def test_accuracy_matches_sklearn_on_wine(wine_data):
    X_train, X_test, y_train, y_test = wine_data

    model = DecisionTreeClassifier(max_depth=5, random_state=0)
    model.fit(X_train, y_train)
    our_acc = model.score(X_test, y_test)

    sklearn_model = SklearnDecisionTreeClassifier(max_depth=5, random_state=0)
    sklearn_model.fit(X_train, y_train)
    sklearn_acc = accuracy_score(y_test, sklearn_model.predict(X_test))

    assert abs(our_acc - sklearn_acc) < 0.05