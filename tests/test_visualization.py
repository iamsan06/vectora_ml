"""
Tests for visualization utilities.
"""

import matplotlib

# Prevent plots from opening during tests
matplotlib.use("Agg")

import pytest

from vectora_ml.utils.datasets import make_regression
from vectora_ml.linear_model import LinearRegression
from vectora_ml.viz import (
    plot_loss,
    plot_regression_line,
    plot_residuals,
)
from vectora_ml.core.exceptions import (
    NotFittedError,
)


@pytest.fixture
def trained_model():
    X, y, _ = make_regression(
        n_samples=100,
        n_features=1,
        noise=5,
        random_state=42,
    )

    model = LinearRegression(
        learning_rate=0.05,
        max_iter=1000,
    )

    model.fit(X, y)

    return model, X, y


def test_plot_loss(trained_model):
    """
    Loss curve should plot without errors.
    """

    model, _, _ = trained_model

    plot_loss(model)


def test_plot_regression_line(trained_model):
    """
    Regression line should plot without errors.
    """

    model, X, y = trained_model

    plot_regression_line(model, X, y)


def test_plot_residuals(trained_model):
    """
    Residual plot should plot without errors.
    """

    model, X, y = trained_model

    plot_residuals(model, X, y)


def test_plot_before_fit():
    """
    Plotting before fit should raise NotFittedError.
    """

    model = LinearRegression()

    with pytest.raises(NotFittedError):
        plot_loss(model)


def test_regression_line_multifeature():
    """
    Regression line only supports one feature.
    """

    X, y, _ = make_regression(
        n_samples=100,
        n_features=3,
        random_state=42,
    )

    model = LinearRegression()

    model.fit(X, y)

    with pytest.raises(ValueError):
        plot_regression_line(model, X, y)