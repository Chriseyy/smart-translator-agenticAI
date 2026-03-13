import os
import sys
import base64
import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from mcp.server.fastmcp import FastMCP

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "src", "document_class_detector", "scripts")
MODELS_DIR = os.path.join(PROJECT_ROOT, "src", "document_class_detector", "models")

sys.path.append(SCRIPTS_DIR)
sys.path.append(MODELS_DIR)

from alexnet import build_model

CKPT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "final_training_alexnet",
    "best.ckpt",
)

mcp = FastMCP("document-class-detector", json_response=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    cfg = ckpt["config"]
    model = build_model(num_classes=cfg["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    class_names = cfg.get("class_names")
    image_size = cfg["image_size"]
    return model, class_names, image_size


model, class_names, image_size = load_checkpoint()


def preprocess_pil(img: Image.Image, image_size: int):
    tf = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return tf(img).unsqueeze(0)


def decode_image(image_b64: str) -> Image.Image:
    img_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


@mcp.tool()
def classify_document(image_b64: str) -> dict:
    img = decode_image(image_b64)
    x = preprocess_pil(img, image_size).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    pred_idx = int(probs.argmax(dim=1).item())
    pred_conf = float(probs[0, pred_idx].item())
    all_probs = probs.squeeze(0).tolist()

    if class_names is not None and 0 <= pred_idx < len(class_names):
        label = str(class_names[pred_idx])
    else:
        label = str(pred_idx)

    return {
        "label": label,
        "class_index": pred_idx,
        "probability": pred_conf,
        "all_probs": all_probs,
    }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
