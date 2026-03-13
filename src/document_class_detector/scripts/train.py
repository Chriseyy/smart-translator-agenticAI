# uv run src/document_class_detector/scripts/train.py --config src/document_class_detector/configs/simple_training.yaml
import argparse
import sys
import os
import json
import yaml
from datetime import datetime
import random

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

# wandb ist optional: Script läuft auch ohne Tracking.
try:
    import wandb
except Exception:
    wandb = None

from data_loader import data_loader

# Projekt-Importpfad ergänzen.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vit import build_model


def parse_args():
    """Parses CLI args and merges them with a YAML config (if provided).

    Returns:
        Parsed argparse Namespace with config values filled in.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    p.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["sgd", "adam"],
        help="Optimizer to use (overrides config if set)",
    )

    # Early stopping (default: patience 5)
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs without improvement). Default=5 if not set.",
    )

    args = p.parse_args()

    # YAML-Config laden und nur Werte setzen, die nicht explizit per CLI gesetzt wurden.
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if not getattr(args, k, None):
                setattr(args, k, v)

    # Default, falls weder CLI noch YAML es setzt.
    if getattr(args, "early_stopping_patience", None) is None:
        args.early_stopping_patience = 5

    return args


def set_seed(seed: int):
    """Sets RNG seeds for reproducible training runs.

    Args:
        seed: Random seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_validation(model, val_dl, device, criterion, num_classes: int):
    """Runs validation and computes loss, accuracy, and macro-F1.

    Args:
        model: PyTorch model to evaluate.
        val_dl: Validation DataLoader.
        device: Torch device.
        criterion: Loss function (e.g. CrossEntropyLoss).
        num_classes: Number of classes (for confusion matrix sizing).

    Returns:
        Tuple (avg_loss, acc, f1_macro). Returns (None, None, None) if val_dl is empty.
    """
    if val_dl is None or len(val_dl.dataset) == 0:
        return None, None, None

    model.eval()
    total_loss, total = 0.0, 0
    correct = 0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    for images, labels in tqdm(val_dl, total=len(val_dl), desc="validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

        # Confusion Matrix effizient updaten via index_add.
        idx = labels * num_classes + preds
        confmat.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    # Macro-F1 aus Confusion Matrix ableiten.
    tp = confmat.diag().to(torch.float32)
    fp = confmat.sum(dim=0).to(torch.float32) - tp
    fn = confmat.sum(dim=1).to(torch.float32) - tp

    denom_p = tp + fp
    denom_r = tp + fn
    precision = torch.where(denom_p > 0, tp / denom_p, torch.zeros_like(tp))
    recall = torch.where(denom_r > 0, tp / denom_r, torch.zeros_like(tp))

    denom_f1 = precision + recall
    f1_per_class = torch.where(
        denom_f1 > 0,
        2 * precision * recall / denom_f1,
        torch.zeros_like(denom_f1),
    )
    f1_macro = f1_per_class.mean().item()

    return avg_loss, acc, f1_macro


def save_checkpoint(path, model, optimizer, epoch, config, best_val_loss, best_val_f1):
    """Saves a training checkpoint with model/optimizer state and metadata.

    Args:
        path: Output checkpoint path.
        model: Model to save.
        optimizer: Optimizer whose state should be saved.
        epoch: Current epoch number.
        config: Config dict stored for reproducibility.
        best_val_loss: Best validation loss so far.
        best_val_f1: Best validation macro-F1 so far.

    Returns:
        None
    """
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    torch.save(payload, path)


def main():
    """Runs supervised training with optional wandb tracking and early stopping."""
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoader bauen.
    train_dl, val_dl, _ = data_loader(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    # Run-Verzeichnis + config dumpen (damit Experimente nachvollziehbar bleiben).
    run_dir = os.path.join(args.ckpt_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Optional: Weights & Biases.
    use_wandb = args.track_with == "wandb"
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb not installed but wandb tracking enabled")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            config=vars(args),
        )

    model = build_model(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer aus args (CLI überschreibt YAML).
    opt_name = getattr(args, "optimizer", "sgd")
    opt_name = opt_name.lower() if isinstance(opt_name, str) else "sgd"

    if opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # -----------------
    # LR SCHEDULER SETUP
    # -----------------
    sched_name = getattr(args, "lr_scheduler", "none").lower()
    warmup_epochs = getattr(args, "warmup_epochs", 0)

    if sched_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - warmup_epochs),
            eta_min=0,
        )
    else:
        scheduler = None

    # -----------------
    # EARLY STOPPING SETUP (basiert auf val_loss)
    # -----------------
    patience = int(getattr(args, "early_stopping_patience", 5))
    best_val_loss = None
    best_val_f1 = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # Warmup: LR linear von 0 -> args.lr rampen.
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warm_factor = epoch / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = args.lr * warm_factor

        model.train()
        train_total_loss = 0.0
        train_total = 0

        pbar = tqdm(train_dl, total=len(train_dl), desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)

            # Wenn hier NaNs/Inf passieren: lieber hart abbrechen.
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()

            # Gradient clipping als Sicherheitsgurt gegen Divergenz.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_total_loss += loss.item() * images.size(0)
            train_total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Scheduler erst nach Warmup aktivieren (falls konfiguriert).
        if scheduler is not None and (warmup_epochs == 0 or epoch > warmup_epochs):
            scheduler.step()

        train_loss = train_total_loss / max(train_total, 1)
        val_loss, val_acc, val_f1 = run_validation(model, val_dl, device, criterion, args.num_classes)

        if val_loss is None:
            print(f"Finished epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val set empty")
        else:
            print(
                f"Finished epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1_macro={val_f1:.4f}"
            )

        if use_wandb:
            log_payload = {"epoch": epoch, "train_loss": train_loss}
            if val_loss is not None:
                log_payload.update({"val_loss": val_loss, "val_acc": val_acc, "val_f1_macro": val_f1})
            log_payload["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(log_payload, step=epoch)

        # Last-Checkpoint für Debug/Resume.
        last_path = os.path.join(run_dir, "last.ckpt")
        save_checkpoint(last_path, model, optimizer, epoch, vars(args), best_val_loss, best_val_f1)

        # -----------------
        # EARLY STOPPING & BEST CHECKPOINT (basiert auf val_loss)
        # -----------------
        if val_loss is not None:
            # Niedriger = besser.
            improved = (best_val_loss is None) or (val_loss < best_val_loss)

            if improved:
                best_val_loss = val_loss
                best_val_f1 = val_f1  # F1 nur als Referenz mitnehmen.
                epochs_no_improve = 0

                best_path = os.path.join(run_dir, "best.ckpt")
                save_checkpoint(best_path, model, optimizer, epoch, vars(args), best_val_loss, best_val_f1)

                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_val_f1"] = best_val_f1
                    wandb.run.summary["best_epoch"] = epoch
            else:
                epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch}: "
                        f"no val_loss improvement for {patience} consecutive epochs "
                        f"(best_val_loss={best_val_loss:.4f}, best_val_f1={best_val_f1:.4f})."
                    )
                    if use_wandb:
                        wandb.run.summary["early_stopped"] = True
                        wandb.run.summary["early_stop_epoch"] = epoch
                        wandb.run.summary["early_stopping_patience"] = patience
                    break

    if use_wandb:
        wandb.finish()

    print(
        f"Training complete. Best model saved with val_loss={best_val_loss:.4f}, "
        f"val_f1={best_val_f1:.4f}"
    )


if __name__ == "__main__":
    main()
