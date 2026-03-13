import os
import sys
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# Projektpfade relativ zur aktuellen Datei auflösen (robust gegen cwd-Wechsel).
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Lokale Module über sys.path verfügbar machen.
sys.path.append(SCRIPTS_DIR)
sys.path.append(MODELS_DIR)

from data_loader import data_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vit import build_model

# Default-Checkpoint-Pfad.
CKPT_PATH = "checkpoints/visionnet/best.ckpt"


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Loads a model checkpoint and returns the model in eval mode plus its config.

    Args:
        ckpt_path: Path to the checkpoint file.
        device: Target device to load the model onto.

    Returns:
        Tuple of (model, cfg) where cfg is the saved config dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(num_classes=cfg["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def get_test_batch(cfg):
    """Fetches a single batch from the test dataloader.

    Args:
        cfg: Configuration dict containing dataset and preprocessing params.

    Returns:
        Tuple of (imgs, labels) from the first test batch.
    """
    _, _, test_dl = data_loader(
        n_train=cfg["n_train"],
        n_val=cfg["n_val"],
        n_test=cfg["n_test"],
        image_size=cfg["image_size"],
        batch_size=100,
    )
    imgs, labels = next(iter(test_dl))
    return imgs, labels


def tensor_to_image_numpy(x: torch.Tensor):
    """Converts a normalized CHW tensor batch (N=1) to an HWC numpy image.

    Args:
        x: Input tensor with shape (1, C, H, W), assumed normalized with mean=0.5/std=0.5.

    Returns:
        HWC numpy array in [0, 1] suitable for matplotlib.
    """
    # Batchdimension entfernen, auf CPU ziehen und von Autograd trennen.
    x = x[0].detach().cpu()
    # Unnormalize: [-1,1] -> [0,1] (passend zur typischen (0.5,0.5)-Normierung).
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    # Für matplotlib: (C,H,W) -> (H,W,C).
    x = x.permute(1, 2, 0).numpy()
    return x


def ig_heatmap_score_entropy(
    model,
    x: torch.Tensor,
    target_idx: int,
    n_steps: int = 50,
    eps: float = 1e-12,
):
    """Computes Integrated Gradients attribution plus simple summary metrics.

    Produces:
      - heatmap: mean attribution over channels, normalized to [0,1]
      - score: mean absolute attribution magnitude
      - entropy: entropy over absolute attributions (flattened)

    Args:
        model: PyTorch model used for attribution.
        x: Input tensor of shape (1, C, H, W).
        target_idx: Class index used as attribution target.
        n_steps: Number of IG steps.
        eps: Small constant for numerical stability.

    Returns:
        Tuple (heatmap, score, entropy) where heatmap is a CPU tensor (H, W).
    """
    # Eigene Leaf-Tensor-Kopie: Captum/Autograd soll sauber daran arbeiten.
    x = x.clone().detach().requires_grad_(True)
    ig = IntegratedGradients(model)
    attr = ig.attribute(x, target=target_idx, n_steps=n_steps)

    # Betrag als "Wichtigkeit" pro Pixel/Channel.
    abs_attr = attr.abs()

    # Simple globale Stärke: Mittelwert über alle Dimensionen.
    score = abs_attr.mean().detach().cpu().item()

    # Entropie über normalisierte Beträge (als Verteilungs-"Spikiness").
    flat = abs_attr.view(-1)
    s = flat.sum()
    if float(s.detach().cpu().item()) == 0.0:
        entropy = float("nan")
    else:
        p = (flat / (s + eps)).clamp_min(eps)
        entropy = float((-p * p.log()).sum().detach().cpu().item())

    # Heatmap pro Pixel: Channel mitteln, dann auf [0,1] skalieren.
    hm = attr.squeeze(0).mean(dim=0)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    heatmap = hm.detach().cpu()

    return heatmap, score, entropy


def safe_mean(xs):
    """Returns the mean of a list, or NaN if the list is empty.

    Args:
        xs: List of numeric values.

    Returns:
        Arithmetic mean or NaN when xs is empty.
    """
    return (sum(xs) / len(xs)) if len(xs) > 0 else float("nan")


def main():
    """Runs IG analysis on a test batch and provides an interactive viewer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Früh scheitern, wenn der Checkpoint fehlt.
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model, cfg = load_checkpoint(CKPT_PATH, device)
    imgs, labels = get_test_batch(cfg)
    imgs = imgs.to(device)
    labels = labels.to(device)

    # Vorhersagen einmal berechnen (ohne Gradients).
    with torch.no_grad():
        logits_all = model(imgs)
        probs_all = F.softmax(logits_all, dim=1)
        pred_all = probs_all.argmax(dim=1)

    correct_mask = (pred_all == labels)

    # Cache, damit beim Durchklicken nicht jedes Mal IG neu gerechnet wird.
    cache = {}

    scores_correct, scores_wrong = [], []
    ent_correct, ent_wrong = [], []

    # Für jedes Sample: IG relativ zur vorhergesagten Klasse berechnen.
    for i in range(len(imgs)):
        x = imgs[i : i + 1]
        pred_idx = int(pred_all[i].item())

        heatmap, score, entropy = ig_heatmap_score_entropy(
            model, x, target_idx=pred_idx, n_steps=50
        )
        cache[i] = (heatmap, score, entropy)

        if bool(correct_mask[i].item()):
            scores_correct.append(score)
            ent_correct.append(entropy)
        else:
            scores_wrong.append(score)
            ent_wrong.append(entropy)

    # Aggregierte Kennzahlen getrennt nach korrekt/falsch.
    mean_abs_ig_correct = safe_mean(scores_correct)
    mean_abs_ig_wrong = safe_mean(scores_wrong)

    entropy_correct = safe_mean([e for e in ent_correct if not math.isnan(e)])
    entropy_wrong = safe_mean([e for e in ent_wrong if not math.isnan(e)])

    delta_error = mean_abs_ig_wrong - mean_abs_ig_correct

    n_correct = int(correct_mask.sum().item())
    n_wrong = int((~correct_mask).sum().item())

    # Interaktive Matplotlib-Ansicht (Pfeiltasten links/rechts).
    fig = plt.figure(figsize=(9, 4.5))
    index = {"i": 0}  # Mutable Container für Closure.

    def update():
        """Redraws the current sample view based on index['i']."""
        fig.clear()
        i = index["i"]
        x = imgs[i : i + 1]

        y_true = int(labels[i].item())
        pred_idx = int(pred_all[i].item())
        pred_conf = float(probs_all[i, pred_idx].item())
        is_correct = bool(correct_mask[i].item())

        heatmap, score, entropy = cache[i]
        img_np = tensor_to_image_numpy(x)

        # Überblick über die Batch-Statistiken + aktuelle Kennzahlen.
        fig.suptitle(
            f"mean(|IG|): correct={mean_abs_ig_correct:.6g} (n={n_correct}) | wrong={mean_abs_ig_wrong:.6g} (n={n_wrong}) | Δ_error={delta_error:.6g}\n"
            f"entropy(|IG|): correct={entropy_correct:.6g} | wrong={entropy_wrong:.6g}",
            fontsize=10,
        )

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img_np)
        ax1.set_title(f"true={y_true}")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img_np)
        ax2.imshow(heatmap.numpy(), cmap="jet", alpha=0.5)
        ax2.set_title(
            f"pred={pred_idx}, p={pred_conf:.2f} | {'OK' if is_correct else 'WRONG'}\n"
            f"mean(|IG|)={score:.6g} | entropy={entropy:.6g}"
        )
        ax2.axis("off")

        fig.canvas.draw_idle()

    def on_key(event):
        """Handles left/right key presses to move through the batch."""
        if event.key == "right":
            index["i"] = (index["i"] + 1) % len(imgs)
            update()
        elif event.key == "left":
            index["i"] = (index["i"] - 1) % len(imgs)
            update()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()


if __name__ == "__main__":
    main()
