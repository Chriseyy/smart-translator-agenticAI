import optuna
import pandas as pd
import numpy as np
import sys
import random
from pathlib import Path
from tqdm import tqdm

# --- PATH SETUP ---
current_file = Path(__file__).resolve()

font_detector_dir = current_file.parent.parent

if str(font_detector_dir) not in sys.path:
    sys.path.append(str(font_detector_dir))

DATA_DIR = font_detector_dir / "data"

if not (DATA_DIR / "font_size_train.csv").exists():
    root_data = font_detector_dir.parent.parent / "data"
    if (root_data / "font_size_train.csv").exists():
        DATA_DIR = root_data
    else:
        print("\nCRITICAL ERROR: Could not find 'font_size_train.csv'.")
        print(f"Checked locations:\n   1. {DATA_DIR}\n   2. {root_data}")
        sys.exit(1)

print(f"Working Directory: {font_detector_dir}")
print(f"Data Directory:    {DATA_DIR}")

try:
    from tiny_diff.preprocessors import normalize_zero_mean_unit_variance
    from tiny_diff.scalar.losses import mse
    from tiny_diff.scalar.node import Node
except ImportError as e:
    print("\nIMPORT ERROR: Could not import 'tiny_diff'.")
    print(f"Python expects the folder 'tiny_diff' inside: {font_detector_dir}")
    print(f"Details: {e}")
    sys.exit(1)

# Model Import
try:
    from model import FontSizeMLP
except ImportError:
    print("\nMPORT ERROR: Could not import 'FontSizeMLP' from model.py")
    sys.exit(1)

# --- Configuration ---
TARGET_FONT = "Arial"
N_TRIALS = 20
EPOCHS_PER_TRIAL = 5
BATCH_SIZE = 200


def prepare_data(df, font_name):
    font_df = df[df["font_name"] == font_name]
    if len(font_df) == 0: return None, None

    # Scale labels by 100.0 as per training logic
    Y = font_df["font_size_label"].values / 100.0
    X = font_df.drop(columns=["font_name", "font_size_label"]).values
    return X, Y


def train_one_epoch(model, X, Y, lr):
    epoch_loss = 0.0
    indices = list(range(len(X)))
    random.shuffle(indices)

    subset_indices = indices[:BATCH_SIZE]
    pbar = tqdm(subset_indices, desc="    Training", leave=False)

    for i in pbar:
        features = [Node(val) for val in X[i]]
        target = [Node(Y[i])]

        pred = model(features)
        loss = mse(pred, target)

        for p in model.parameters(): p.grad = 0.0
        loss.backward()

        for p in model.parameters():
            if p.grad > 1.0: p.grad = 1.0
            if p.grad < -1.0: p.grad = -1.0
            p.value -= lr * p.grad

        epoch_loss += loss.value
        pbar.set_postfix({"loss": f"{loss.value:.4f}"})

    return epoch_loss / len(subset_indices)


def evaluate(model, X, Y):
    total_loss = 0.0
    indices = list(range(len(X)))
    subset = indices[:100]

    for i in subset:
        features = [Node(val) for val in X[i]]
        target = [Node(Y[i])]
        pred = model(features)
        loss = mse(pred, target)
        total_loss += loss.value
    return total_loss / len(subset)


def objective(trial):
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    train_df = pd.read_csv(DATA_DIR / "font_size_train.csv")
    val_df = pd.read_csv(DATA_DIR / "font_size_val.csv")

    X_train_raw, Y_train = prepare_data(train_df, TARGET_FONT)
    X_val_raw, Y_val = prepare_data(val_df, TARGET_FONT)

    X_train, mean, std = normalize_zero_mean_unit_variance(X_train_raw)
    X_val = (X_val_raw - mean) / (std + 1e-8)

    model = FontSizeMLP(input_dim=X_train.shape[1])

    epoch_pbar = tqdm(range(EPOCHS_PER_TRIAL), desc=f"Trial {trial.number}", leave=False)
    val_loss = float('inf')

    for epoch in epoch_pbar:
        train_loss = train_one_epoch(model, X_train, Y_train, learning_rate)
        val_loss = evaluate(model, X_val, Y_val)

        trial.report(val_loss, epoch)
        epoch_pbar.set_postfix({"val_loss": f"{val_loss:.4f}", "lr": f"{learning_rate:.1e}"})

        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


def main():
    print(f"Starting Optuna optimization for {TARGET_FONT}...")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")

    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\nOptimization aborted.")

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    if len(study.trials) > 0:
        print(f"Best Learning Rate: {study.best_params['lr']:.6f}")
        print(f"Best Loss:          {study.best_value:.6f}")
        print("-" * 50)
        print(f"Update 'scripts/train_font_size.py' with LEARNING_RATE = {study.best_params['lr']:.5f}")
    else:
        print("No trials completed.")
    print("=" * 50)


if __name__ == "__main__":
    main()