import numpy as np
import pytest
from vectora_ml.core import BaseEstimator
from vectora_ml.core.exceptions import NotFittedError

class DummyEstimator(BaseEstimator):

    def __init__(self, learning_rate=0.1):
        super().__init__()
        self.learning_rate = learning_rate

    def fit(self, X, y=None):
        self.coef_ = np.array([1.0])
        self._mark_fitted()
        return self

    def predict(self, X):
        self._check_is_fitted("coef_")
        return np.zeros(len(X))

def test_get_params():

    model = DummyEstimator(learning_rate=0.5)

    params = model.get_params()

    assert params["learning_rate"] == 0.5

def test_set_params():

    model = DummyEstimator()

    model.set_params(learning_rate=1.5)

    assert model.learning_rate == 1.5

def test_not_fitted():

    model = DummyEstimator()

    with pytest.raises(NotFittedError):
        model.predict(np.array([[1], [2]]))

def test_fit_sets_attribute():

    model = DummyEstimator()

    model.fit(np.array([[1], [2]]))

    assert hasattr(model, "coef_")

def test_predict():

    model = DummyEstimator()

    model.fit(np.array([[1], [2]]))

    predictions = model.predict(np.array([[3], [4]]))

    assert len(predictions) == 2

def test_mark_fitted():

    model = DummyEstimator()

    assert model._is_fitted is False

    model.fit(np.array([[1], [2]]))

    assert model._is_fitted is True