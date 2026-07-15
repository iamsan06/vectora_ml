"""
Custom exception classes used throughout Vectora-ML.

These exceptions provide clear, user-friendly error messages
instead of generic Python exceptions.
"""


class VectoraMLError(Exception):
    """
    Base exception for all Vectora-ML specific errors.

    Every custom exception in the library should inherit
    from this class.
    """

    pass


class NotFittedError(VectoraMLError):
    """
    Raised when an estimator is used before calling fit().
    """

    pass


class DimensionMismatchError(VectoraMLError):
    """
    Raised when input arrays have incompatible dimensions.
    """

    pass


class InvalidParameterError(VectoraMLError):
    """
    Raised when a parameter value is invalid.
    """

    pass


class ConvergenceError(VectoraMLError):
    """
    Raised when an optimization algorithm fails
    to converge.
    """

    pass