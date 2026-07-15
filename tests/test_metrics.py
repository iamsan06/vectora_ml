import numpy as np

from vectora_ml.core.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def test_mse():

    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])

    assert np.isclose(mean_squared_error(y_true, y_pred), 1 / 3)


def test_rmse():

    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])

    assert np.isclose(root_mean_squared_error(y_true, y_pred), np.sqrt(1 / 3))


def test_mae():

    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])

    assert np.isclose(mean_absolute_error(y_true, y_pred), 1 / 3)


def test_r2():

    y_true = np.array([1, 2, 3])

    assert np.isclose(r2_score(y_true, y_true), 1.0)


def test_accuracy():

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    assert np.isclose(accuracy_score(y_true, y_pred), 0.75)


def test_precision():

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0])

    assert np.isclose(precision_score(y_true, y_pred), 2 / 3)


def test_recall():

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0])

    assert np.isclose(recall_score(y_true, y_pred), 1.0)


def test_f1():

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0])

    precision = 2 / 3
    recall = 1.0
    expected = 2 * precision * recall / (precision + recall)

    assert np.isclose(f1_score(y_true, y_pred), expected)