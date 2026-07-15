"""
Evaluation metrics for machine learning models.
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Compute Root Mean Squared Error.
    """

    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Compute coefficient of determination.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Compute Binary Cross-Entropy (log loss).

    This is the loss function LogisticRegression trains on. Unlike
    MSE, it heavily penalizes confident-but-wrong predictions
    (e.g. predicting 0.99 for a sample whose true label is 0).

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).

    y_pred : array-like
        Predicted probabilities, in (0, 1).

    eps : float
        Small constant to avoid log(0), which would be -inf.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip predictions so log() never sees exactly 0 or 1.
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )


def accuracy_score(y_true, y_pred):
    """
    Classification accuracy.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    """
    Binary precision.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    """
    Binary recall.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """
    Binary F1 score.
    """

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)