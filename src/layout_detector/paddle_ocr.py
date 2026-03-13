"""PaddleOCR runner and simple CLI.

Overview
- Applies small LangChain shims (via `utils.apply_paddlex_langchain_shims`) so
    importing `paddleocr` works even when older `paddlex` expects legacy modules.
- Provides a single entry-point `run_ocr(image_path, save_dir)` that runs
    PaddleOCR and saves two artifacts for the first result:
        * `ocr_result.jpg` — visualization image from the model result
        * `ocr_result.json` — JSON serialization of the model result
- Returns a compact JSON-style dict with paths and a status:
        {"status": "success"|"error", "input_path": str,
         "ocr_image": str|None, "ocr_json": str|None, "error": str|None}

Usage
- As a library:
        from src.layout_detector.paddle_ocr import run_ocr
        info = run_ocr("path/to/img.jpg")
        print(info)

- As a CLI:
    uv run python -m src.layout_detector.paddle_ocr path/to/img.jpg --out data/layout_detector

Notes
- Only the first page/result is saved; extend if you need more.
- The function focuses on saving model-native outputs (image + JSON). If you
    want custom overlays or additional fields, add a separate renderer.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .utils import apply_paddlex_langchain_shims

# Disable oneDNN (MKL-DNN) BEFORE importing paddle/paddleocr.
# PaddlePaddle 3.x built with oneDNN crashes at inference with:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>] (onednn_instruction.cc:116)
# Direct assignment is required — setdefault() is silently ignored by the
# compiled C++ inference engine once it has already been initialised.
import os as _os
_os.environ["FLAGS_use_mkldnn"] = "0"
_os.environ["PADDLE_DISABLE_MKLDNN"] = "1"
_os.environ["FLAGS_enable_pir_in_executor"] = "0"

# Apply shims BEFORE importing paddleocr (which pulls paddlex)
apply_paddlex_langchain_shims()

from paddleocr import PaddleOCR, DocPreprocessor
try:
    import paddle  # optional: presence check only
except Exception:  # pragma: no cover - not strictly required
    paddle = None

PROJECT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
)

# Default output location used by the CLI and by `run_ocr` when no save_dir is given.
OUTPUT_DIR = os.path.join(PROJECT_DATA_DIR, "layout_detector")

# Only used for the optional CLI default argument; may not exist in all setups.
DEFAULT_IMAGE_PATH = os.path.join(PROJECT_DATA_DIR, "layout_detector", "images", "test3.jpeg")

MAX_SIDE_LIMIT = 4000  # PaddleOCR default max side; resize bigger inputs proactively

_OCR_SINGLETON: Optional[PaddleOCR] = None


def _patch_paddle_disable_mkldnn() -> None:
    """Monkey-patch paddle.inference.Config to disable oneDNN on every predictor.

    PaddlePaddle 3.x compiled with oneDNN ignores FLAGS_use_mkldnn env vars and
    crashes at inference. Intercepting Config.__init__ and calling
    disable_mkldnn() there is the only reliable workaround for pre-built wheels.
    """
    try:
        import paddle.inference as _pi
        _OrigConfig = _pi.Config

        class _NoDnnConfig(_OrigConfig):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                try:
                    self.disable_mkldnn()
                except Exception:
                    pass

        _pi.Config = _NoDnnConfig
    except Exception:
        pass  # If paddle.inference is unavailable, env vars are the best we can do


def _get_ocr_instance() -> PaddleOCR:
    """Return a cached PaddleOCR instance to avoid repeated heavy allocations."""
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        _patch_paddle_disable_mkldnn()
        _OCR_SINGLETON = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _OCR_SINGLETON


def run_ocr(image_path: str, save_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    """Run PaddleOCR on an image and return a JSON-style dict.

    The returned dict contains:
      status: "success" or "error"
      input_path: original image path provided
    ocr_image: path to saved OCR visualization image (or None)
    ocr_json: path to saved OCR JSON (or None)
    error: optional error message when status == "error"

    Side effects:
      - Saves first result's visualization image as `ocr_result.jpg`
      - Saves first result's JSON serialization as `ocr_result.json`
    Args:
        image_path: Path to the input image to process.
        save_dir: Base directory where OCR artifacts are saved.
    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - status (str): "success" on OCR completion, otherwise "error"
            - input_path (str): Original image path
            - ocr_image (str|None): Path to rendered OCR visualization
            - ocr_json (str|None): Path to OCR JSON output
            - error (str|None): Error message if OCR failed
    """
    out_dir = os.path.join(save_dir, "ocr")
    os.makedirs(out_dir, exist_ok=True)

    response: Dict[str, Any] = {
        "status": "error",
        "input_path": image_path,
        "ocr_image": None,
        "ocr_json": None,
    }

    if not os.path.exists(image_path):
        response["error"] = f"Input image not found: {image_path}"
        return response

    # Resize very large images in-memory before OCR to avoid PaddleOCR max_side_limit errors.
    ocr_input: Any = image_path
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            max_dim = max(width, height)
            if max_dim > MAX_SIDE_LIMIT:
                scale = MAX_SIDE_LIMIT / float(max_dim)
                new_w = max(1, int(width * scale))
                new_h = max(1, int(height * scale))
                resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                img = img.resize((new_w, new_h), resample=resample)
            # PaddleOCR accepts numpy arrays directly; avoid writing temp files
            ocr_input = np.array(img)
    except Exception as e_resize:
        response["error"] = f"Resize/prep failed: {e_resize}"
        return response

    try:
        import glob as _glob
        ocr = _get_ocr_instance()
        results = ocr.predict(ocr_input)
        if not results:
            response["error"] = "No OCR results returned"
            return response
        first = results[0]

        # Save visualization image
        # PaddleOCR 3.x may silently rename the output file, so fall back to a
        # glob scan and rename if the expected path is absent.
        img_path = os.path.join(out_dir, "ocr_result.jpg")
        try:
            first.save_to_img(img_path)
            if not os.path.exists(img_path):
                candidates = _glob.glob(os.path.join(out_dir, "*.jpg")) + \
                             _glob.glob(os.path.join(out_dir, "*.png"))
                if candidates:
                    os.rename(max(candidates, key=os.path.getmtime), img_path)
            if os.path.exists(img_path):
                response["ocr_image"] = img_path
        except Exception as e_img:
            response["error"] = f"Failed saving OCR image: {e_img}"

        # Save JSON
        json_path = os.path.join(out_dir, "ocr_result.json")
        try:
            first.save_to_json(json_path)
            if not os.path.exists(json_path):
                candidates = _glob.glob(os.path.join(out_dir, "*.json"))
                if candidates:
                    os.rename(max(candidates, key=os.path.getmtime), json_path)
            if os.path.exists(json_path):
                response["ocr_json"] = json_path
            else:
                prev = response.get("error")
                response["error"] = (prev + "; " if prev else "") + \
                                     "save_to_json produced no output file"
        except Exception as e_json:
            prev = response.get("error")
            response["error"] = (prev + f"; {e_json}" if prev else f"Failed saving OCR json: {e_json}")

        if response["ocr_image"] or response["ocr_json"]:
            response["status"] = "success"
        elif "error" not in response:
            response["error"] = "Artifacts not saved"

        return response

    except Exception as e:
        response["error"] = f"OCR execution failed: {e}"
        return response


if __name__ == "__main__":  # manual test example
    import argparse
    parser = argparse.ArgumentParser(description="Run PaddleOCR and return paths JSON")
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE_PATH, help="Path to input image")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Base output directory")
    args = parser.parse_args()
    result_json = run_ocr(args.image, save_dir=args.out)
    print(json.dumps(result_json, indent=2))