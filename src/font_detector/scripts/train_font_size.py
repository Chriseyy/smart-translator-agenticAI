import pandas as pd
import numpy as np
import pickle
import random
from pathlib import Path
from tqdm import tqdm
import sys
import wandb

current_script_path = Path(__file__).resolve()

font_detector_dir = current_script_path.parent.parent
sys.path.append(str(font_detector_dir))

project_root = font_detector_dir.parent.parent
sys.path.append(str(project_root))

from tiny_diff.preprocessors import normalize_zero_mean_unit_variance
from tiny_diff.scalar.losses import mse
from tiny_diff.scalar.node import Node
from tiny_diff.scalar.io import save_model_weights

from src.font_detector.model import FontSizeMLP

DATA_DIR = Path("data")
MODELS_DIR = Path("src/font_detector/models/font_size")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# List of the fonts that are being trained
FONTS_TO_TRAIN = ["Arial", "Times New Roman", "Verdana", "Courier New", "Comic Sans MS"]

# Hyperparameters
LEARNING_RATE = 0.05026
EPOCHS = 20


def prepare_data(df, font_name):
    font_df = df[df["font_name"] == font_name]

    if len(font_df) == 0:
        return None, None

    # Target: Font size scaled (e.g. 12pt -> 0.12)
    Y = font_df["font_size_label"].values
    Y = Y / 100.0

    drop_cols = ["font_name", "font_size_label"]
    X = font_df.drop(columns=drop_cols).values

    return X, Y


def evaluate(model, X, Y):
    """Calculates mean squared error on the validation set."""
    total_loss = 0.0
    for i in range(len(X)):
        features = [Node(val) for val in X[i]]
        target = [Node(Y[i])]

        prediction = model(features)
        loss = mse(prediction, target)
        total_loss += loss.value

    return total_loss / len(X)


def train_sgd_epoch(model, X_norm, Y, lr):
    epoch_loss = 0.0
    n_samples = len(X_norm)
    indices = list(range(n_samples))
    random.shuffle(indices)

    pbar = tqdm(indices, desc="  Training", leave=False)

    for step, i in enumerate(pbar):
        features = [Node(val) for val in X_norm[i]]
        target = [Node(Y[i])]

        prediction = model(features)
        loss = mse(prediction, target)

        # Reset gradients
        for p in model.parameters(): p.grad = 0.0

        # Backward pass
        loss.backward()

        # Gradient Clipping & Update
        for p in model.parameters():
            if p.grad > 1.0: p.grad = 1.0
            if p.grad < -1.0: p.grad = -1.0
            p.value -= lr * p.grad

        epoch_loss += loss.value

        # Update progress bar
        if step % 200 == 0:
            real_pred = prediction[0].value * 100
            real_target = target[0].value * 100
            pbar.set_description(f"Loss: {loss.value:.4f} | Target: {real_target:.1f}pt -> Pred: {real_pred:.1f}pt")

    return epoch_loss / n_samples


def main():
    print("--- Start Training ---")

    if not (DATA_DIR / "font_size_train.csv").exists():
        print(f"Error: No data found in {DATA_DIR}.")
        return

    print("Load CSV Files...")
    train_df = pd.read_csv(DATA_DIR / "font_size_train.csv")
    val_df = pd.read_csv(DATA_DIR / "font_size_val.csv")

    for font_name in FONTS_TO_TRAIN:
        print(f"\n{'=' * 40}")
        print(f"TRAINING FOR FONT: {font_name}")
        print(f"{'=' * 40}")

        # Initialize WandB run for this specific font
        wandb.init(
            project="font-size-detector",
            group="multi-font-training",
            name=f"train_{font_name}",
            config={
                "font": font_name,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "architecture": "MLP"
            },
            reinit=True
        )

        X_train_raw, Y_train = prepare_data(train_df, font_name)
        X_val_raw, Y_val = prepare_data(val_df, font_name)

        if X_train_raw is None:
            print(f"-> Skipping {font_name} (No data found)")
            wandb.finish()
            continue

        # Normalization
        X_train, mean, std = normalize_zero_mean_unit_variance(X_train_raw)
        epsilon = 1e-8
        X_val = (X_val_raw - mean) / (std + epsilon)

        scaler_path = MODELS_DIR / f"{font_name}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump({"mean": mean, "std": std}, f)
        print(f"-> Scaler saved: {scaler_path.name}")

        input_dim = X_train.shape[1]
        model = FontSizeMLP(input_dim)
        print(f"-> Model created (Input Dim: {input_dim})")

        print(f"-> Start Training ({EPOCHS} Epochs, {len(X_train)} Samples)...")

        for epoch in range(EPOCHS):
            train_loss = train_sgd_epoch(model, X_train, Y_train, lr=LEARNING_RATE)

            # Validation step for logging
            val_loss = evaluate(model, X_val, Y_val)

            print(f"   Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Log metrics to WandB
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": LEARNING_RATE
            })

        model_filename = f"{font_name}_model.json"
        save_path = MODELS_DIR / model_filename

        save_model_weights(model, str(save_path))
        print(f"-> Model finished and saved: {model_filename}")

        # Finish the run for this font so the next loop starts a fresh one
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except ImportError as e:
        print(f"\nImport Error: {e}")