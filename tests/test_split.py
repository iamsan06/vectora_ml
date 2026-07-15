import numpy as np

from vectora_ml.utils.preprocessing import (
    train_test_split,
)


def test_train_test_split():

    X = np.arange(100).reshape(50, 2)

    y = np.arange(50)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    assert len(X_train) == 40
    assert len(X_test) == 10

    assert len(y_train) == 40
    assert len(y_test) == 10