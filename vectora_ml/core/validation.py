"""
Core-level validation helpers.

These live in `core` (not `utils`) because BaseEstimator depends on
them directly. Keeping them here means `core` never needs to import
anything from `utils`, which keeps the dependency graph one-directional:
utils -> core, never core -> utils.
"""


def check_is_fitted(estimator, attributes):
    """
    Verify that an estimator has been fitted.

    Parameters
    ----------
    estimator

    attributes : str or list[str]
        Required fitted attributes.
    """

    if isinstance(attributes, str):
        attributes = [attributes]

    missing = [
        attr
        for attr in attributes
        if not hasattr(estimator, attr)
    ]

    if missing:
        from vectora_ml.core.exceptions import NotFittedError

        raise NotFittedError(
            f"{estimator.__class__.__name__} is not fitted yet. "
            f"Missing attributes: {missing}"
        )