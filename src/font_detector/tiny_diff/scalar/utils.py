import numpy as np
from typing import List, Union
from tiny_diff.scalar.node import Node


def np_array_to_nodes(X: Union[List[float], List[List[float]], np.ndarray]) -> List[List[Node]]:
    """
    Convert a 1D or 2D array-like input into a list of lists of Node objects.

    Each scalar value becomes a Node with an optional name for tracking.

    Args:
        X: Input data. Can be:
            - 1D list of floats
            - 2D list of floats
            - 1D or 2D NumPy array

    Returns:
        List[List[Node]]: Nested list of Nodes.
            - For 1D input, each element is wrapped in a single-element list.
            - For 2D input, each row is a list of Nodes.

    Raises:
        ValueError: If X has more than 2 dimensions.
    """
    X = np.array(X, dtype=float)  # ensure a NumPy array

    if X.ndim == 1:
        # Convert 1D array to list of single-element lists of Nodes
        return [[Node(val, name=f"x{j}")] for j, val in enumerate(X)]
    elif X.ndim == 2:
        # Convert 2D array to list of lists of Nodes
        X_nodes: List[List[Node]] = []
        for i, row in enumerate(X):
            X_nodes.append([Node(val, name=f"x{i}_{j}") for j, val in enumerate(row)])
        return X_nodes
    else:
        raise ValueError("X.ndim must be 1 or 2")
