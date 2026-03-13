import pandas as pd
import numpy as np
import pickle
import sys
import random
from pathlib import Path
from tqdm import tqdm

current_script_path = Path(__file__).resolve()

font_detector_dir = current_script_path.parent.parent
sys.path.append(str(font_detector_dir))

project_root = font_detector_dir.parent.parent
sys.path.append(str(project_root))

from tiny_diff.scalar.node import Node
from tiny_diff.scalar.io import load_model_weights
from src.font_detector.model import FontSizeMLP

DATA_DIR = Path("data")
MODELS_DIR = Path("src/font_detector/models/font_size")

FONTS_TO_TEST = ["Arial", "Times New Roman", "Verdana", "Courier New", "Comic Sans MS"]


def evaluate_font(font_name, df):
    """
    Loads the model and scaler for a specific font, performs predictions,
    and calculates the Mean Absolute Error (MAE).
    """
    print(f"\n{'=' * 40}")
    print(f"EVALUATION: {font_name}")
    print(f"{'=' * 40}")

    font_df = df[df["font_name"] == font_name]
    if len(font_df) == 0:
        print(f"[Warning] No test data found for {font_name}. Skipping.")
        return

    scaler_path = MODELS_DIR / f"{font_name}_scaler.pkl"
    model_path = MODELS_DIR / f"{font_name}_model.json"

    if not scaler_path.exists() or not model_path.exists():
        print(f"[Error] Model or Scaler for {font_name} not found.")
        print("Please run the training script first.")
        return

    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
        mean = scaler_data["mean"]
        std = scaler_data["std"]

    Y_true = font_df["font_size_label"].values

    drop_cols = ["font_name", "font_size_label"]
    X_raw = font_df.drop(columns=drop_cols).values

    epsilon = 1e-8
    X_norm = (X_raw - mean) / (std + epsilon)

    input_dim = X_norm.shape[1]
    model = FontSizeMLP(input_dim)

    load_model_weights(model, str(model_path))
    print(f"-> Model loaded. Predicting on {len(X_norm)} samples...")

    predictions = []
    absolute_errors = []

    for i in tqdm(range(len(X_norm)), desc="  Testing", leave=False):
        features = [Node(val) for val in X_norm[i]]

        out_node = model(features)

        # Inverse Scaling:
        # In training, we did Y / 100.0. Here we multiply by 100 to get real 'pt' size.
        pred_size = out_node[0].value * 100.0
        true_size = Y_true[i]

        predictions.append(pred_size)
        absolute_errors.append(abs(pred_size - true_size))

    mae = np.mean(absolute_errors)
    max_error = np.max(absolute_errors)
    std_error = np.std(absolute_errors)

    print(f"-> Mean Absolute Error (MAE): {mae:.4f} pt")
    print(f"-> Max Error: {max_error:.4f} pt")
    print(f"-> Std Dev of Error: {std_error:.4f} pt")

    print("\n   [Examples: True Size vs Prediction]")
    indices = np.random.choice(len(predictions), size=min(3, len(predictions)), replace=False)
    for idx in indices:
        diff = predictions[idx] - Y_true[idx]
        print(f"   - Target: {Y_true[idx]:.2f} pt | Pred: {predictions[idx]:.2f} pt | Diff: {diff:+.2f}")


def main():
    print("--- Start Evaluation ---")


    test_csv_path = DATA_DIR / "font_size_test.csv"

    if not test_csv_path.exists():
        print(f"[Info] Test file '{test_csv_path}' not found.")
        print("[Info] Falling back to Validation set...")
        test_csv_path = DATA_DIR / "font_size_val.csv"

    if not test_csv_path.exists():
        print(f"[Error] No data found in {DATA_DIR}.")
        return

    print(f"Load CSV Data from: {test_csv_path}")
    df = pd.read_csv(test_csv_path)

    for font in FONTS_TO_TEST:
        evaluate_font(font, df)

    print("\n--- Evaluation Finished ---")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Check if 'tiny_diff' and 'src' are correctly in your PYTHONPATH.")