from typing import List
from tiny_diff.scalar.node import Node, log, softmax

def cross_entropy(prediction: List[Node], target_idx: int) -> Node:
    """
    Compute the cross-entropy loss for a single target class.

    Args:
        prediction (list of Node): Predicted probabilities for each class.
        target_idx (int): Index of the true class.

    Returns:
        Node: Cross-entropy loss.
    """
    # Negative log-likelihood for the target class
    return -log(prediction[target_idx])


def mse(prediction: List[Node], target: List[Node]) -> Node:
    """
    Compute Mean Squared Error (MSE) between predictions and targets.

    Args:
        prediction (list of Node): Predicted values.
        target (list of Node): Target/true values.

    Returns:
        Node: Sum of squared differences.
    """
    assert len(prediction) == len(target), "Prediction and target must have the same length"

    # Calculate squared error for each pair
    squared_errors =  [(p-t) ** 2 for p, t in zip(prediction, target)]

    # Sum up all squared errors
    total_error = sum(squared_errors)

    # Calculate the Mean error
    return total_error * (1.0 / len(prediction))

