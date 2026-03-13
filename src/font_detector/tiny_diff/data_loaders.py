import csv
import numpy as np
from typing import Tuple, List, Optional

def load_fontsize_csv(
    csv_file: str,
    n_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load features, targets, and texts from a CSV file.

    The CSV is expected to have numeric columns for features, with the last
    column as the target. Optionally, limit the number of samples read.

    Args:
        csv_file (str): Path to the CSV file.
        n_samples (int, optional): Maximum number of rows to read.

    Returns:
        Tuple containing:
            - X (np.ndarray): Feature array of shape (n_samples, n_features).
            - Y (np.ndarray): Target array of shape (n_samples, ).
            - texts (List[str]): Original last column values as strings.
    """
    X_list: List[List[float]] = []
    Y_list: List[float] = []
    texts: List[str] = []

    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)  # skip header

        for i, row in enumerate(reader):
            if n_samples is not None and i >= n_samples:
                break

            # Convert all columns to float except last
            row_values: List[float] = [float(v) for v in row[:-1]]
            # Features = all columns except last numeric column
            features: List[float] = row_values[:-1]
            # Target = last numeric column
            target: float = row_values[-1]

            texts.append(row[-1])  # keep original string of last column
            X_list.append(features)
            Y_list.append(target)

    X: np.ndarray = np.array(X_list, dtype=np.float32)
    Y: np.ndarray = np.array(Y_list, dtype=np.float32)

    return X, Y, texts
