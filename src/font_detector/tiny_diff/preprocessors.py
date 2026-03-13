import numpy as np
from typing import Tuple


def normalize_zero_mean_unit_variance(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize a 2D NumPy array to have zero mean and unit variance for each feature.

    Args:
        X (np.ndarray): Input array of shape (n_samples, n_features).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X_norm: Normalized array with zero mean and unit variance.
            - mean: Mean of each feature in the original array.
            - std: Standard deviation of each feature in the original array.

    Notes:
        A small epsilon (1e-8) is added to std to prevent division by zero.
    """

    # Calculate mean for each feature
    mean = np.mean(X, axis=0)

    # Calculate standard deviation (std) for each feature
    std = np.std(X, axis=0)

    # Define epsilon to prevent division by zero
    epsilon = 1e-8

     # Normalize each feature by subtracting mean and dividing by std
    X_norm = (X - mean) / (std + epsilon)

    return X_norm, mean, std