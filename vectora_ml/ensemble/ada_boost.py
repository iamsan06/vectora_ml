"""
AdaBoost Classifier — sequential decision stumps with sample reweighting.
"""

import numpy as np

from vectora_ml.core.estimator import BaseEstimator
from vectora_ml.core.exceptions import (
    DimensionMismatchError,
    InvalidParameterError,
)
from vectora_ml.core.metrics import accuracy_score
from vectora_ml.utils.validation import check_array, check_X_y
from vectora_ml.tree.decision_tree import DecisionTreeClassifier


class AdaBoostClassifier(BaseEstimator):
    """
    AdaBoost for binary classification (the classic AdaBoost.M1
    algorithm), using decision stumps (depth-1 trees) as weak learners.

    The core idea: train a weak learner, see which samples it got
    wrong, upweight those samples so the NEXT weak learner is forced to
    focus on them, and combine every weak learner into a weighted vote
    where more accurate learners get more say. Each round only needs to
    do slightly better than a coin flip — the boosting is what turns a
    sequence of "weak" learners into a strong one.

    Internally, labels are mapped to {-1, +1} (the classic AdaBoost
    formulation, where the sign of a prediction IS its vote) and mapped
    back to {0, 1} in predict().

    Parameters
    ----------
    n_estimators : int, default=50
        Number of weak learners (decision stumps) to train.

    learning_rate : float, default=1.0
        Shrinks each weak learner's vote weight (alpha_m). Smaller
        values need more estimators but tend to generalize better —
        the same shrinkage/n_estimators trade-off as Gradient Boosting.

    max_depth : int, default=1
        Depth of each weak learner. 1 (a decision stump — a single
        split) is the classic AdaBoost choice; weak learners are meant
        to be only slightly better than random guessing.
    """

    def __init__(self, n_estimators=50, learning_rate=1.0, max_depth=1):
        super().__init__()

        if n_estimators <= 0:
            raise InvalidParameterError(
                "n_estimators must be a positive integer."
            )

        if learning_rate <= 0:
            raise InvalidParameterError("learning_rate must be positive.")

        if max_depth <= 0:
            raise InvalidParameterError(
                "max_depth must be a positive integer."
            )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Train the ensemble via the classic AdaBoost.M1 reweighting loop:
        fit a weighted stump, score how wrong it was, weight its vote
        by how right it was, then reweight samples for the next round.
        """

        X, y = check_X_y(X, y)

        unique_labels = np.unique(y)

        if not np.all(np.isin(unique_labels, [0, 1])):
            raise InvalidParameterError(
                "AdaBoostClassifier only supports binary targets "
                "encoded as 0 and 1."
            )

        n_samples = X.shape[0]

        # Classic AdaBoost works with {-1, +1} labels rather than
        # {0, 1} — the sign of a prediction then directly IS its vote,
        # which is what makes the weighted-vote combination clean.
        y_signed = np.where(y == 1, 1, -1)

        # Every sample starts equally important.
        sample_weight = np.full(n_samples, 1 / n_samples)

        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        for m in range(self.n_estimators):

            stump = DecisionTreeClassifier(max_depth=self.max_depth)
            stump.fit(X, y, sample_weight=sample_weight)

            predictions = stump.predict(X)
            predictions_signed = np.where(predictions == 1, 1, -1)

            incorrect = predictions_signed != y_signed

            # Weighted error rate: how much of the sample WEIGHT (not
            # just count) this stump got wrong.
            weighted_error = np.sum(sample_weight[incorrect]) / np.sum(sample_weight)

            # Clip away from exactly 0 or 1 so log() below never blows
            # up — a stump can occasionally get everything right (or,
            # rarely, everything wrong) on a given weighted round.
            weighted_error = np.clip(weighted_error, 1e-10, 1 - 1e-10)

            # Weak learner weight: how much say this stump gets in the
            # final vote. A near-perfect stump (error -> 0) gets a huge
            # alpha; a coin-flip stump (error = 0.5) gets alpha = 0 (no
            # say at all); worse-than-random (error > 0.5) gets negative
            # alpha, which just flips its vote — even an "anti-expert"
            # is useful if you listen to the opposite of what it says.
            alpha = self.learning_rate * 0.5 * np.log(
                (1 - weighted_error) / weighted_error
            )

            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(weighted_error)

            # Reweight: samples this stump got WRONG get their weight
            # multiplied by e^alpha (upweighted, so the next round has
            # to pay attention to them); samples it got RIGHT get
            # multiplied by e^-alpha (downweighted). Renormalize so the
            # weights sum back to 1.
            sample_weight = sample_weight * np.exp(
                -alpha * y_signed * predictions_signed
            )
            sample_weight = sample_weight / np.sum(sample_weight)

        self._mark_fitted()

        return self

    def decision_function(self, X):
        """
        Return each sample's total weighted vote (sum of
        alpha_m * prediction_m across all stumps). Positive leans
        toward class 1, negative toward class 0 — magnitude reflects
        how confidently the ensemble agrees.
        """

        self._check_is_fitted(
            ["estimators_", "estimator_weights_", "n_features_in_"]
        )

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise DimensionMismatchError(
                f"Expected {self.n_features_in_} features, but got {X.shape[1]}."
            )

        vote_total = np.zeros(X.shape[0])

        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            predictions = stump.predict(X)
            predictions_signed = np.where(predictions == 1, 1, -1)
            vote_total += alpha * predictions_signed

        return vote_total

    def predict(self, X):
        """
        Predict class labels (0 or 1) via the sign of the weighted vote.
        """

        vote_total = self.decision_function(X)

        return (vote_total >= 0).astype(int)

    def score(self, X, y):
        """
        Return the classification accuracy of the ensemble on the given data.
        """

        y_pred = self.predict(X)

        return accuracy_score(np.asarray(y), y_pred)