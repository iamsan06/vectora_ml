"""
Example: Plot the training loss of Linear Regression.
"""

from vectora_ml.linear_model import LinearRegression
from vectora_ml.utils.datasets import make_regression
from vectora_ml.viz import plot_loss


def main():
    # Generate a synthetic regression dataset
    X, y, _ = make_regression(
        n_samples=200,
        n_features=1,
        noise=15,
        random_state=42,
    )

    # Train the model
    model = LinearRegression(
        learning_rate=0.05,
        max_iter=1000,
    )

    model.fit(X, y)

    print(f"Training completed in {model.n_iter_} iterations")
    print(f"Final loss: {model.final_loss_:.6f}")

    # Visualize training loss
    plot_loss(model)


if __name__ == "__main__":
    main()