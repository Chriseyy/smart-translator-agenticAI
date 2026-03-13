# uv run src/document_class_detector/scripts/hyperparameter_opti.py --config src/document_class_detector/configs/simple_training.yaml --trials 100
import argparse
import os
import sys
import json
import yaml
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from urllib.parse import urlparse

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

# wandb ist optional: Script läuft auch ohne Tracking.
try:
    import wandb
except Exception:
    wandb = None

from data_loader import data_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vit import build_model
from train import run_validation, set_seed

# import torch.multiprocessing as mp
# mp.set_sharing_strategy("file_system")
# mp.set_start_method("spawn", force=True)


def _ensure_sqlite_dir(url: str):
    """Ensures the directory for a SQLite Optuna storage URL exists.

    Args:
        url: Optuna storage URL (e.g. sqlite:////abs/path/to.db).

    Returns:
        None
    """
    # Nur relevant für SQLite (file-based), nicht für In-Memory.
    if not url or not url.startswith("sqlite"):
        return
    if url.startswith("sqlite:///:memory:"):
        return

    # urlparse liefert Pfad inkl. führendem "/" (bei sqlite:///...).
    p = urlparse(url).path
    if p.startswith("/"):
        p = p[1:]
    dirpath = os.path.dirname(os.path.abspath(p))
    os.makedirs(dirpath, exist_ok=True)


def load_base_args(cfg_path: str):
    """Loads a YAML config and exposes keys as attributes.

    Args:
        cfg_path: Path to YAML config file.

    Returns:
        Simple object with config entries accessible via dot notation.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Minimaler Wrapper, damit man args.foo statt cfg["foo"] schreiben kann.
    class Obj:
        def __init__(self, d):
            self.__dict__.update(d)

    return Obj(cfg)


def build_run_dir(base_ckpt_dir: str, exp_name: str, trial_number: int) -> str:
    """Creates and returns a per-trial run directory.

    Args:
        base_ckpt_dir: Base directory for checkpoints.
        exp_name: Experiment name namespace.
        trial_number: Optuna trial number.

    Returns:
        Absolute/relative path to the created trial directory.
    """
    # Jede Trial bekommt einen eigenen Ordner (logs, ckpts, config).
    run_dir = os.path.join(base_ckpt_dir, exp_name, f"trial_{trial_number:04d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_checkpoint(path, model, optimizer, epoch, config, best_val_f1):
    """Saves a training checkpoint with model/optimizer state and metadata.

    Args:
        path: Output checkpoint path.
        model: Trained model.
        optimizer: Optimizer instance.
        epoch: Current epoch number.
        config: Config dict to store for reproducibility.
        best_val_f1: Best validation F1 so far.

    Returns:
        None
    """
    # Einheitliches Payload-Format, damit sich Runs später sauber vergleichen lassen.
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "best_val_f1": best_val_f1,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    torch.save(payload, path)


def objective(trial: optuna.Trial, base_cfg_path: str, use_wandb_default: bool):
    """Optuna objective: trains one trial and returns the best validation macro-F1.

    Args:
        trial: Optuna trial handle (suggest/report/prune).
        base_cfg_path: YAML config path providing non-tuned defaults.
        use_wandb_default: Whether wandb tracking is enabled by default.

    Returns:
        Best validation macro-F1 achieved in this trial (float).
    """
    # Base-Config laden und dann Trial-spezifisch überschreiben.
    args = load_base_args(base_cfg_path)

    # --- Hyperparameter-Suchraum ---
    args.lr = trial.suggest_float("lr", 0.001, 0.3, log=True)
    args.momentum = trial.suggest_float("momentum", 0.6, 0.99)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 0.01, log=True)
    args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    # ------------------------------

    print(f"\n=== Trial {trial.number} ===")
    print(
        f"optimizer={optimizer_name}, lr={args.lr:.5f}, momentum={args.momentum:.3f}, "
        f"weight_decay={args.weight_decay:.6f}, batch_size={args.batch_size}"
    )

    # Pro Trial anderer Seed, aber reproduzierbar über base_seed.
    base_seed = int(getattr(args, "seed", 42))
    set_seed(base_seed + trial.number)

    device_str = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str if device_str in ("cpu", "cuda") else "cpu")

    # DataLoader mit Trial-Batchsize bauen.
    train_dl, val_dl, _ = data_loader(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    # Trial-Ordner + Trial-Config dumpen.
    run_dir = build_run_dir(args.ckpt_dir, args.exp_name, trial.number)
    with open(os.path.join(run_dir, "trial_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Optional: wandb Tracking (nur wenn explizit so konfiguriert).
    use_wandb = use_wandb_default and getattr(args, "track_with", None) == "wandb"
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb missing but tracking enabled")
        wandb.init(
            project=getattr(args, "wandb_project", None),
            entity=getattr(args, "wandb_entity", None),
            name=f"{args.exp_name}-t{trial.number}",
            config={**vars(args), "trial_number": trial.number},
            reinit=True,
        )

    model = build_model(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # --- Optimizer-Auswahl ---
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:  # adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    # ------------------------

    # Early Stopping pro Trial.
    patience = getattr(args, "early_stop_patience", 5)
    min_delta = getattr(args, "early_stop_min_delta", 0)
    no_improve = 0
    best_val_f1 = None
    best_epoch = 0
    best_path = os.path.join(run_dir, "best.ckpt")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        # Train Loop mit tqdm, damit man bei vielen Trials Feedback bekommt.
        pbar = tqdm(
            train_dl,
            total=len(train_dl),
            desc=f"[t{trial.number}] epoch {epoch}/{args.epochs}",
            leave=False,
        )
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(seen, 1)

        # Val-Metriken zentral über run_validation.
        val_loss, val_acc, val_f1 = run_validation(
            model, val_dl, device, criterion, args.num_classes
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1_macro": val_f1,
                    "trial_number": trial.number,
                },
                step=epoch,
            )

        # Last-Checkpoint (für Debug/Resume) immer schreiben.
        last_path = os.path.join(run_dir, "last.ckpt")
        save_checkpoint(last_path, model, optimizer, epoch, vars(args), best_val_f1)

        # Improvement-Check gegen best_val_f1 mit min_delta.
        improved = (
            (best_val_f1 is None and val_f1 is not None)
            or (
                val_f1 is not None
                and best_val_f1 is not None
                and (val_f1 - best_val_f1) > min_delta
            )
        )

        if improved:
            best_val_f1 = float(val_f1)
            best_epoch = epoch
            save_checkpoint(best_path, model, optimizer, epoch, vars(args), best_val_f1)
            if use_wandb:
                wandb.run.summary["best_val_f1"] = best_val_f1
                wandb.run.summary["best_epoch"] = best_epoch
            no_improve = 0
        else:
            no_improve += 1
            if use_wandb:
                wandb.log({"early_stop_no_improve": no_improve}, step=epoch)

        # Optuna-Reporting pro Epoch (für Pruning/Plots).
        metric_for_optuna = float(val_f1) if val_f1 is not None else 0.0
        trial.report(metric_for_optuna, step=epoch)
        if trial.should_prune():
            if use_wandb:
                wandb.finish()
            raise optuna.TrialPruned()

        # Early Stop, wenn zu lange keine Verbesserung.
        if patience is not None and no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if use_wandb:
        wandb.finish()

    # Optional: best.ckpt nochmal laden (stellt sicher, dass best wirklich best ist).
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    return float(best_val_f1 if best_val_f1 is not None else 0.0)


def parse_cli():
    """Parses command-line arguments for running the Optuna study.

    Returns:
        Parsed argparse Namespace.
    """
    # Minimal-CLI: config + Optuna-Settings.
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--study-name", type=str, default="alexnet_hpo")
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    """Entry point: creates/loads the Optuna study, runs optimization, and exports results."""
    args = parse_cli()

    # Default-Storage: lokale SQLite DB unter checkpoints/.
    if args.storage is None:
        db_path = os.path.abspath("checkpoints/hyper_learbing/optuna_rvl_alexnet.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        args.storage = f"sqlite:///{db_path}"

    _ensure_sqlite_dir(args.storage)
    print(f"[INFO] Storage: {args.storage}")

    # Sampler/Pruner: TPE + Successive Halving.
    sampler = TPESampler(seed=args.seed, n_startup_trials=30, multivariate=True)
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Wenn schon Trials existieren: letzten Params einmal neu enqueuen (Stabilitäts-Check).
    trials = study.get_trials(deepcopy=False)
    if trials:
        last = trials[-1]
        study.enqueue_trial(last.params)
        print(
            f"[ENQUEUE] Re-running params from last trial #{last.number} (state={last.state})."
        )

    print(f"[INFO] Study '{args.study_name}' geladen, bisher {len(study.trials)} Trials.")
    print(f"[INFO] Neue Trials werden bis {args.trials} ausgeführt.\n")

    study.optimize(
        lambda tr: objective(tr, args.config, use_wandb_default=(not args.no_wandb)),
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # Ergebnisse als CSV exportieren (praktisch für schnelle Auswertung).
    df = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "intermediate_values")
    )
    os.makedirs("checkpoints/hyper_learbing", exist_ok=True)
    out_csv = os.path.join("checkpoints", "hyper_learbing", f"{args.study_name}_trials.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nTrials-CSV gespeichert als: {out_csv}")

    # Best Trial kurz in die Konsole drucken.
    best = study.best_trial
    print("\n=== Best Trial ===")
    print(f"Trial #{best.number}")
    print(f"Value (best val_f1_macro): {best.value:.6f}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
