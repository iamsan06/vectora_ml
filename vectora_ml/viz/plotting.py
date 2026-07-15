import numpy as np
import matplotlib.pyplot as plt
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
)

def _check_loss_history(model):
    """
    Ensure the model has been trained and contains loss history.
    """

    model._check_is_fitted(
        [
            "coef_",
            "intercept_",
            "loss_history_",
        ]
    )

    if len(model.loss_history_) == 0:
        raise ValueError(
            "Model does not contain any recorded training loss."
        )

def plot_loss(model, figsize=(8, 5)):
    """
    Plot the training loss over iterations.

    Parameters
    ----------
    model
        Trained estimator containing ``loss_history_``.

    figsize : tuple
        Figure size.
    """

    _check_loss_history(model)

    plt.figure(figsize=figsize)

    plt.plot(
        range(1, len(model.loss_history_) + 1),
        model.loss_history_,
        linewidth=2,
    )

    plt.title(f"{model.__class__.__name__} Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_regression_line(
    model,
    X,
    y,
    figsize=(8, 5),
):
    """
    Plot a fitted regression line.

    Only works for one-dimensional regression.
    """

    model._check_is_fitted(
        ["coef_", "intercept_", "n_features_in_"]
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise DimensionMismatchError(
            "Expected X to be a 2D array."
        )

    if X.shape[1] != 1:
        raise ValueError(
            "Regression line visualization only supports "
            "one input feature."
        )

    predictions = model.predict(X)

    order = np.argsort(X[:, 0])

    X_sorted = X[order]
    pred_sorted = predictions[order]

    plt.figure(figsize=figsize)

    plt.scatter(
        X[:, 0],
        y,
        label="Samples",
        alpha=0.8,
    )

    plt.plot(
        X_sorted[:, 0],
        pred_sorted,
        linewidth=2,
        label="Regression Line",
    )

    plt.title("Linear Regression Fit")
    plt.xlabel("Feature")
    plt.ylabel("Target")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_residuals(
    model,
    X,
    y,
    figsize=(8, 5),
):
    """
    Plot residuals versus predicted values.
    """

    model._check_is_fitted(
        ["coef_", "intercept_"]
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    predictions = model.predict(X)

    residuals = y - predictions

    plt.figure(figsize=figsize)

    plt.scatter(
        predictions,
        residuals,
        alpha=0.8,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
    )

    plt.title("Residual Plot")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")

    plt.grid(True)

    plt.tight_layout()
    plt.show()