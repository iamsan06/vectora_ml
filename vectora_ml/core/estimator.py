"""
Base estimator class.

All machine learning models in Vectora-ML inherit from this class.
"""

from abc import ABC, abstractmethod
from vectora_ml.core.validation import check_is_fitted


class BaseEstimator(ABC):
    """
    Base class for every estimator.
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, X, y=None):
        """
        Train the estimator.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict outputs.
        """
        pass

    def get_params(self):
        """
        Return constructor parameters.
        """

        params = {}

        for key, value in self.__dict__.items():

            if key.startswith("_"):
                continue

            if key.endswith("_"):
                continue

            params[key] = value

        return params

    def set_params(self, **params):
        """
        Set estimator parameters.
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def _mark_fitted(self):
        """
        Mark estimator as trained.
        """

        self._is_fitted = True

    def _check_is_fitted(self, attributes):
        """
        Ensure estimator has been fitted.
        """

        check_is_fitted(self, attributes)