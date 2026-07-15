"""
Node used internally by DecisionTreeClassifier to represent tree structure.
"""


class Node:
    """
    A single node in a decision tree.

    Internal (decision) nodes store a feature_index/threshold used to
    route samples, plus two children. Leaf nodes store a predicted
    `value` and leave everything else as None.
    """

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        """
        A node is a leaf if and only if it holds a predicted value.
        """

        return self.value is not None