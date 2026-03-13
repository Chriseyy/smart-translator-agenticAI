# uv run ./src/document_class_detector/scripts/test_model.py --ckpt-path checkpoints/restnet_50/best.ckpt
import argparse
import sys
import os
import csv

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from data_loader import data_loader

# Projekt-Imports über relative Pfade ermöglichen.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.alexnet import build_model


def parse_args():
    """Parses CLI arguments for test set evaluation.

    Returns:
        Parsed argparse Namespace.
    """
    # Minimal-CLI: Checkpoint + Testset-Größe + Output für Fehlklassifikationen.
    p = argparse.ArgumentParser("Minimal evaluation on test set")
    p.add_argument("--ckpt-path", type=str, required=True)
    p.add_argument("--n-test", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=16)
    p.add_argument("--miscls-outdir", type=str, default="misclassifications")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, criterion, miscls_outdir: str):
    """Evaluates a model on a dataloader and logs misclassifications.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader for the test set.
        device: Torch device to run inference on.
        criterion: Loss function (e.g. CrossEntropyLoss).
        miscls_outdir: Directory where misclassification CSV is written.

    Returns:
        Tuple (avg_loss, acc, cm, total_images, skipped_batches).
    """
    model.eval()
    total_loss, total = 0.0, 0
    correct = 0

    all_preds = []
    all_labels = []

    skipped_batches = 0

    # Output-Verzeichnis + CSV initialisieren.
    os.makedirs(miscls_outdir, exist_ok=True)
    miscls_csv_path = os.path.join(miscls_outdir, "misclassifications.csv")
    with open(miscls_csv_path, "w", newline="") as miscls_f:
        miscls_w = csv.writer(miscls_f)
        miscls_w.writerow(["path", "true", "pred"])

        it = iter(loader)
        pbar = tqdm(total=len(loader), desc="test", leave=True)

        # Optional: versuche Sample-Pfade über dataset.index zu bekommen.
        ds = getattr(loader, "dataset", None)
        index_list = getattr(ds, "index", None)

        sample_i = 0  # globaler Offset in die Indexliste

        while True:
            try:
                images, labels = next(it)
            except StopIteration:
                break
            except Exception as e:
                # Robust gegen sporadische DataLoader-Probleme.
                skipped_batches += 1
                pbar.update(1)
                pbar.set_postfix(skipped_batches=skipped_batches)
                print(f"[WARN] Skipping a batch due to DataLoader error: {type(e).__name__}: {e}")
                continue

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            bs = images.size(0)

            total_loss += loss.item() * bs
            correct += (preds == labels).sum().item()
            total += bs

            # Für Confusion Matrix später alles sammeln.
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # Fehlklassifikationen batchweise in CSV schreiben.
            labels_1d = labels.detach().view(-1)
            preds_1d = preds.detach().view(-1)

            mis_mask = preds_1d.ne(labels_1d)
            if mis_mask.any():
                mis_idx = mis_mask.nonzero(as_tuple=False).view(-1).tolist()
                for b in mis_idx:
                    true_c = int(labels_1d[b].item())
                    pred_c = int(preds_1d[b].item())

                    # Pfad rekonstruieren, falls dataset.index verfügbar ist.
                    path = ""
                    if index_list is not None:
                        gi = sample_i + b
                        if gi < len(index_list):
                            path = index_list[gi][0]

                    miscls_w.writerow([str(path), str(true_c), str(pred_c)])

                # Flush, damit bei langen Runs nichts verloren geht.
                miscls_f.flush()

            sample_i += bs

            # Live-Metriken im Progressbar.
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{(total_loss / max(total, 1)):.4f}",
                acc=f"{(correct / max(total, 1)):.4f}",
                total_images=total,
                skipped_batches=skipped_batches,
            )

        pbar.close()

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    # Confusion Matrix nur bauen, wenn überhaupt Samples durchgelaufen sind.
    if total == 0:
        cm = np.zeros((0, 0), dtype=np.int64)
    else:
        all_preds_np = torch.cat(all_preds).numpy()
        all_labels_np = torch.cat(all_labels).numpy()
        cm = confusion_matrix(all_labels_np, all_preds_np)

    return avg_loss, acc, cm, total, skipped_batches


def plot_confusion_matrix(cm, out_path):
    """Plots and saves a confusion matrix image.

    Args:
        cm: Confusion matrix as a 2D numpy array.
        out_path: File path for the saved PNG.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)

    # Werte als Text-Overlay (hell/dunkel je nach Hintergrund).
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    """Loads a checkpoint, runs test evaluation, and writes reports/artifacts."""
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Checkpoint laden + num_classes bevorzugt aus der gespeicherten config nehmen.
    ckpt = torch.load(args.ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    num_classes = cfg.get("num_classes", args.num_classes)

    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Test-Loader bauen (train/val dummy klein halten).
    _, _, test_dl = data_loader(
        n_train=1,
        n_val=1,
        n_test=args.n_test,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, cm, total_images, skipped_batches = evaluate(
        model, test_dl, device, criterion, args.miscls_outdir
    )

    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    print(f"total_images={total_images}")
    print(f"skipped_batches={skipped_batches}")
    print(f"misclassifications_saved_to={os.path.join(args.miscls_outdir, 'misclassifications.csv')}")

    # Confusion Matrix nur schreiben, wenn etwas ausgewertet wurde.
    if cm.size > 0:
        plot_confusion_matrix(cm, "confusion_matrix.png")
    else:
        print("[WARN] No valid samples processed; confusion_matrix.png not written.")


if __name__ == "__main__":
    main()
