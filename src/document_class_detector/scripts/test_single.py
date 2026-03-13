# Usage:
# uv run ./src/document_class_detector/scripts/test_single.py --ckpt-path checkpoints/restnet_50/best.ckpt --image-path data/rvl-cdip/images/imagesr/r/g/e/rge31d00/503210033+-0034.tif

import argparse
import os
import sys

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Projekt-Importpfad ergänzen.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.resnet_50 import build_model


def parse_args():
    """Parses CLI arguments for single-image inference.

    Returns:
        Parsed argparse Namespace.
    """
    p = argparse.ArgumentParser("Infer single image")
    p.add_argument("--ckpt-path", type=str, required=True)
    p.add_argument("--image-path", type=str, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=16)
    p.add_argument("--class-names", type=str, default=None)
    return p.parse_args()


def load_class_names(path: str):
    """Loads class names from a text file (one label per line).

    Args:
        path: Path to the class-names file. If None, returns None.

    Returns:
        List of class names, or None if not provided/empty.
    """
    if path is None:
        return None

    # Leere Zeilen rausfiltern, damit Indizes sauber passen.
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines()]
    names = [n for n in names if n]
    return names if len(names) > 0 else None


def build_preprocess_like_dataloader(image_size: int):
    """Builds the same preprocessing pipeline as the training dataloader.

    Args:
        image_size: Target center-crop size.

    Returns:
        Torchvision transform callable.
    """
    return transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


@torch.no_grad()
def infer_one(model, image_path: str, device, preprocess, class_names=None):
    """Runs inference on a single image and returns prediction + full probability vector.

    Args:
        model: Trained PyTorch model.
        image_path: Path to the input image.
        device: Torch device to run on.
        preprocess: Transform pipeline producing a normalized tensor.
        class_names: Optional list mapping class index -> name.

    Returns:
        Tuple (pred_idx, pred_name, pred_prob, probs_np).
    """
    # Bild laden und auf Modell-Input (1,C,H,W) bringen.
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)

    model.eval()
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred_prob = float(probs[pred_idx].item())

    # Optional: sprechender Klassenname, falls vorhanden.
    pred_name = None
    if class_names is not None and pred_idx < len(class_names):
        pred_name = class_names[pred_idx]

    return pred_idx, pred_name, pred_prob, probs.detach().cpu().numpy()


def main():
    """Loads a checkpoint and performs inference on a single image."""
    args = parse_args()

    # Input-Checks, damit Fehler sofort klar sind.
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt_path}")
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"image not found: {args.image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Checkpoint laden + num_classes bevorzugt aus gespeicherter config nehmen.
    ckpt = torch.load(args.ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    num_classes = cfg.get("num_classes", args.num_classes)

    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])

    class_names = load_class_names(args.class_names)
    preprocess = build_preprocess_like_dataloader(args.image_size)

    pred_idx, pred_name, pred_prob, probs = infer_one(
        model, args.image_path, device, preprocess, class_names
    )

    # Hauptprediction ausgeben.
    if pred_name is None:
        print(f"pred_class={pred_idx} prob={pred_prob:.6f}")
    else:
        print(f"pred_class={pred_idx} ({pred_name}) prob={pred_prob:.6f}")

    # Top-k Übersicht (default k=5).
    topk = min(5, probs.shape[0])
    top_idx = np.argsort(-probs)[:topk]
    print("topk:")
    for i in top_idx:
        name = class_names[i] if class_names is not None and i < len(class_names) else str(i)
        print(f"  {i}: {name}  prob={probs[i]:.6f}")


if __name__ == "__main__":
    main()
