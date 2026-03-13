import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import timm

from captum.attr import Occlusion

# Projektpfade relativ zur aktuellen Datei auflösen (robust gegen cwd-Wechsel).
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Lokale Module über sys.path verfügbar machen.
sys.path.append(SCRIPTS_DIR)
sys.path.append(MODELS_DIR)

from data_loader import data_loader

# Default-Checkpoint-Pfad.
CKPT_PATH = "checkpoints/visionnet/best.ckpt"


def build_model(num_classes: int) -> nn.Module:
    """Builds a ViT model and replaces the classification head.

    Args:
        num_classes: Number of output classes.

    Returns:
        A ViT model with a custom linear head.
    """
    # Timm-ViT laden und Head passend zur Klassenzahl ersetzen.
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Loads a checkpoint and returns the model in eval mode plus its config.

    Args:
        ckpt_path: Path to the checkpoint file.
        device: Target device.

    Returns:
        Tuple of (model, cfg) where cfg is the saved config dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(num_classes=cfg["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def get_test_batch(cfg, batch_size=64):
    """Fetches a single batch from the test dataloader.

    Args:
        cfg: Configuration dict containing dataset parameters.
        batch_size: Unused here (kept for API compatibility).

    Returns:
        Tuple (imgs, labels, filenames). filenames can be None.
    """
    # data_loader liefert (train, val, test); hier nur test nutzen.
    _, _, test_dl = data_loader(
        n_train=cfg["n_train"],
        n_val=cfg["n_val"],
        n_test=cfg["n_test"],
        image_size=cfg["image_size"],
        batch_size=50,  # Achtung: überschreibt den Funktionsparameter.
    )
    batch = next(iter(test_dl))

    # Defensive: akzeptiere (imgs, labels) oder (imgs, labels, filenames).
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        imgs, labels = batch[0], batch[1]
        filenames = batch[2] if len(batch) >= 3 else None
    else:
        raise RuntimeError("Unexpected batch format from data_loader")

    return imgs, labels, filenames


def tensor_to_image_numpy(x: torch.Tensor):
    """Converts a normalized CHW tensor batch (N=1) to an HWC numpy image.

    Args:
        x: Input tensor with shape (1, C, H, W), assumed normalized with mean=0.5/std=0.5.

    Returns:
        HWC numpy array in [0, 1] suitable for matplotlib.
    """
    # Batchdimension entfernen, auf CPU ziehen und von Autograd trennen.
    x = x[0].detach().cpu()
    # Unnormalize: [-1,1] -> [0,1].
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    # Für matplotlib: (C,H,W) -> (H,W,C).
    x = x.permute(1, 2, 0).numpy()
    return x


def normalize_2d_map(m: torch.Tensor) -> torch.Tensor:
    """Normalizes a 2D saliency/attribution map to [0, 1].

    Args:
        m: Attribution map tensor (H, W) or (1, H, W).

    Returns:
        Normalized map with values in [0, 1].
    """
    # Falls noch eine Batchdimension dran hängt: weg damit.
    if m.dim() == 3:
        m = m[0]
    mn = m.amin()
    mx = m.amax()
    return (m - mn) / (mx - mn + 1e-8)


def main():
    """Runs occlusion attribution on a test batch and provides an interactive viewer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Früh scheitern, wenn der Checkpoint fehlt.
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model, cfg = load_checkpoint(CKPT_PATH, device)
    imgs, labels, filenames = get_test_batch(cfg, batch_size=64)
    imgs = imgs.to(device)
    labels = labels.to(device)

    print("Model class:", model.__class__.__name__)

    # Vorhersagen einmal berechnen (ohne Gradients).
    with torch.no_grad():
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        pred_all = probs.argmax(dim=1)

    # Captum-Occlusion Setup + Cache, damit man flüssig durchklicken kann.
    occ = Occlusion(model)
    occ_cache = {}

    # Occlusion-Parameter: Fenster über Input schieben.
    WINDOW = (3, 8, 8)
    STRIDE = (3, 4, 4)
    BASELINE = 0.0

    def get_occ_map(i: int) -> torch.Tensor:
        """Computes (or loads) a normalized occlusion map for sample i.

        Args:
            i: Index into the current batch.

        Returns:
            Normalized occlusion map tensor on CPU with shape (H, W).
        """
        if i in occ_cache:
            return occ_cache[i]

        x = imgs[i : i + 1]
        target = int(pred_all[i].item())  # Zielklasse: aktuelle Vorhersage.

        # Occlusion selbst braucht keine Gradients im üblichen Sinn.
        with torch.no_grad():
            attr = occ.attribute(
                x,
                target=target,
                sliding_window_shapes=WINDOW,
                strides=STRIDE,
                baselines=BASELINE,
            )

        # Channels zusammenfassen: pro Pixel eine Wichtigkeit.
        m = attr.abs().sum(dim=1).detach().cpu()
        m = normalize_2d_map(m)
        occ_cache[i] = m
        return m

    # Interaktive Matplotlib-Ansicht (Pfeiltasten links/rechts).
    fig = plt.figure(figsize=(8.5, 4.5))
    index = {"i": 0}  # Mutable Container für Closure.

    def update():
        """Redraws the current sample view based on index['i']."""
        fig.clear()
        i = index["i"]

        x = imgs[i : i + 1]
        img_np = tensor_to_image_numpy(x)
        y_true = int(labels[i].item())
        pred_idx = int(pred_all[i].item())
        pred_conf = float(probs[i, pred_idx].item())

        hm = get_occ_map(i).numpy()

        # Optional: Dateiname oben anzeigen, falls vorhanden.
        if filenames is not None:
            try:
                fig.suptitle(str(filenames[i]), fontsize=10)
            except Exception:
                pass

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img_np)
        ax1.set_title(f"true={y_true}")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img_np)
        ax2.imshow(hm, cmap="jet", alpha=0.5)
        ax2.set_title(f"pred={pred_idx}, p={pred_conf:.2f} | Occlusion")
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
