"""DocTR-based document outline detector.

This module provides a drop-in `find_document_outline` API compatible with the
layout-detector MCP wrapper in `layout_detector.py`.

High-level approach
- Run DocTR OCR to detect text regions.
- Use the hull/min-area rectangle around detected text blocks as a proxy for
    the document boundary.
- Warp the image via a perspective transform.
- Optionally apply a conservative auto-rotation step based on OCR word-box
    geometry.

Outputs
- By default (`output_dir="data"`), results are written under the project-level
    `data/` directory (next to `src/`).
- The extracted image is saved as `<base>_document.jpg` in `output_dir`.

If you need the older contour/debug-image pipeline, see
`outline_detector_old_version.py`.
"""

import os
import json
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
from doctr.models import ocr_predictor

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)

# Use the same helper functions as above for consistency
def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points into TL, TR, BR, BL sequence for stable warps.

    Processing order:
        1. Use coordinate sums to find TL/BR
        2. Use coordinate diffs to find TR/BL
        3. Return ordered 4x2 array

    Args:
        pts: Array of shape (4, 2) containing four corner points.

    Returns:
        np.ndarray: Ordered points shaped (4, 2) as TL, TR, BR, BL.

    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def transform_document(image: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]], np.ndarray, Tuple[int, int]]:
    """
    Warp a quadrilateral document region to a top-down view.

    Produces the warped image plus metadata needed to reproject coordinates back
    into the original image space.

    Processing order:
        1. Order quad points
        2. Compute destination rectangle and homography
        3. Apply perspective warp

    Args:
        image: Input BGR image containing the document.
        quad: (4, 2) array of document corner points.

    Returns:
        Tuple[np.ndarray, List[Dict[str, float]], np.ndarray, Tuple[int, int]]:
            - warped image
            - ordered corner metadata
            - homography matrix
            - warp size (width, height)
    """
    pts = quad.reshape(4, 2).astype("float32")
    ordered = _order_points(pts)
    coordinate_labels = ["top_left", "top_right", "bottom_right", "bottom_left"]
    coordinates_struct = [{"label": lbl, "x": float(pt[0]), "y": float(pt[1])} for lbl, pt in zip(coordinate_labels, ordered)]

    (tl, tr, br, bl) = ordered
    maxWidth = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, coordinates_struct, M, (maxWidth, maxHeight)


def _doctr_text_score(out) -> tuple[int, int, float]:
    """
    Summarize OCR output for rotation scoring.

    Computes counts of horizontal words, total words, and confidence sum to
    guide orientation selection.

    Processing order:
        1. Iterate pages/blocks/lines/words
        2. Accumulate total words and confidence
        3. Count words with width >= height

    Args:
        out: DocTR predictor output object.

    Returns:
        tuple[int, int, float]: (horizontal_words, total_words, sum_confidence).

    Raises:
        None.
    """
    horizontal_words = 0
    total_words = 0
    conf_sum = 0.0

    for page in out.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    total_words += 1
                    if hasattr(word, "confidence") and word.confidence is not None:
                        conf_sum += float(word.confidence)

                    geom = getattr(word, "geometry", None)
                    if geom is None:
                        continue
                    (a, b) = geom  # normalized (xmin, ymin), (xmax, ymax)
                    ww = float(b[0] - a[0])
                    hh = float(b[1] - a[1])
                    if ww >= hh:
                        horizontal_words += 1

    return horizontal_words, total_words, conf_sum


def _auto_rotate_upright_bgr(
    image_bgr: np.ndarray,
    model,
    *,
    min_words: int = 5,
    min_horizontal_ratio: float = 0.55,
    min_score_gain: float = 0.12,
) -> np.ndarray:
    """
    Select the best of four right-angle rotations using OCR yield as signal.

    Scores each rotation by word count, confidence, and horizontal layout bias,
    choosing the winner only if it clearly beats the runner-up to avoid
    ambiguous flips.

    Processing order:
        1. Generate 0/90/180/270 rotations
        2. Run DocTR on each and score
        3. Pick the best score with ambiguity guards

    Args:
        image_bgr: Input image in BGR format.
        model: Preloaded DocTR predictor.
        min_words: Minimum detected words required to rotate.
        min_horizontal_ratio: Minimum horizontal-word ratio to accept a rotation.
        min_score_gain: Minimum relative score gain over runner-up to accept.

    Returns:
        np.ndarray: Upright-rotated image (or original if ambiguous/insufficient).
    """

    def score_rotation(img: np.ndarray, name: str) -> Dict[str, Any]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = model([rgb])
        horizontal_words, total_words, conf_sum = _doctr_text_score(out)
        ratio = (horizontal_words / total_words) if total_words > 0 else 0.0
        avg_conf = conf_sum / max(total_words, 1)
        # Reward text quantity, confidence, and a mild bias toward horizontal layout
        score = (conf_sum + total_words) * (0.70 + 0.30 * ratio) + avg_conf * 5.0
        return {
            "name": name,
            "image": img,
            "total_words": total_words,
            "horizontal_ratio": ratio,
            "conf_sum": conf_sum,
            "avg_conf": avg_conf,
            "score": score,
        }

    rotations = [
        (image_bgr, "0"),
        (cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), "90_cw"),
        (cv2.rotate(image_bgr, cv2.ROTATE_180), "180"),
        (cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), "90_ccw"),
    ]

    scored = [score_rotation(img, name) for img, name in rotations]
    scored_sorted = sorted(scored, key=lambda it: it["score"], reverse=True)
    best = scored_sorted[0]
    runner_up = scored_sorted[1] if len(scored_sorted) > 1 else None

    if best["total_words"] < min_words:
        return image_bgr
    if best["horizontal_ratio"] < min_horizontal_ratio and runner_up is not None and runner_up["score"] > best["score"]:
        return image_bgr
    if runner_up is not None and (best["score"] - runner_up["score"]) < (min_score_gain * max(1.0, runner_up["score"])):
        return image_bgr

    return best["image"]


def _min_area_rect_from_points(points: np.ndarray, expand_ratio: float = 1.0) -> np.ndarray:
    """
    Compute a (possibly expanded) minimum-area rectangle around points.

    Processing order:
        1. Fit min-area rect via cv2
        2. Expand width/height by the given ratio
        3. Return box points

    Args:
        points: Point cloud as (N, 2) array.
        expand_ratio: Multiplicative expansion for rect size.

    Returns:
        np.ndarray: Four box points of the expanded rectangle.

    """
    rect = cv2.minAreaRect(points.astype(np.float32))
    (cx, cy), (rw, rh), angle = rect
    rw = max(1.0, rw * float(expand_ratio))
    rh = max(1.0, rh * float(expand_ratio))
    rect = ((cx, cy), (rw, rh), angle)
    return cv2.boxPoints(rect).astype(np.float32)

def _find_document_doctr(
    img_path: str,
    output_dir: str = "data",
    model=None,
    text_hull_expand_ratio: float = 1.05,
    auto_upright: bool = True,
) -> Dict[str, Any]:
    """
    Run the DocTR-based document detector pipeline.

    Uses OCR text boxes to derive a document hull, warp the page, and optionally
    auto-rotate to upright. Saves the warped page and metadata to disk.

    Processing order:
        1. Resolve output directory and load model if needed
        2. Read image and run DocTR OCR
        3. Build text hull and min-area rectangle with margin
        4. Warp document and optionally auto-rotate
        5. Save warped output and metadata

    Args:
        img_path: Path to the input image.
        output_dir: Where to write `<base>_document.jpg` ("data" resolves to project data/).
        model: Optional preloaded DocTR predictor; loads pretrained if None.
        text_hull_expand_ratio: Expansion factor for the text hull rectangle.
        auto_upright: Whether to attempt 0/90/180/270 rotation selection.

    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - status (str): "success" or "error"
            - message (str): Status or error detail
            - coordinates (list|None): Corner metadata
            - document_image (str|None): Saved warped image path
            - homography (list|None): 3x3 homography matrix serialized
            - warp_size (dict|None): Width/height of the warp
            - destination_corners (list|None): Target rectangle corners

    Raises:
        RuntimeError: Propagated only if caller chooses to raise separately.
    """
    result = {
        "status": "error",
        "message": "Failed",
        "coordinates": None,
        "document_image": None,
        "homography": None,
        "warp_size": None,
        "destination_corners": None,
    }

    # Keep simple: default "data" means project-level data/ next to src/
    if output_dir == "data":
        output_dir = OUTPUT_DIR
    elif not os.path.isabs(output_dir):
        project_root = os.path.normpath(os.path.join(OUTPUT_DIR, ".."))
        output_dir = os.path.normpath(os.path.join(project_root, output_dir))
    
    if model is None:
        model = ocr_predictor(pretrained=True)

    image = cv2.imread(img_path)
    if image is None: return result
    h, w = image.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Inference
    # DocTR's predictor expects actual images (ndim == 3), not file path strings
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out = model([rgb_image])
    
    # Collect all text bounding box corners to find the page boundary
    points = []
    for page in out.pages:
        for block in page.blocks:
            # DocTR uses normalized (0 to 1) [xmin, ymin, xmax, ymax]
            (a, b) = block.geometry
            points.append([a[0] * w, a[1] * h])
            points.append([b[0] * w, b[1] * h])
            points.append([a[0] * w, b[1] * h])
            points.append([b[0] * w, a[1] * h])

    if not points:
        result["message"] = "No text detected to define document area"
        return result

    # Min-area rectangle around text hull (with a bit more margin)
    pts = np.array(points, dtype=np.float32)
    box = _min_area_rect_from_points(pts, expand_ratio=text_hull_expand_ratio)

    try:
        warped, coords, H, warp_size = transform_document(image, box)
        if auto_upright and model is not None:
            warped = _auto_rotate_upright_bgr(warped, model)
        # Match outline_detector.py output naming
        out_path = os.path.join(output_dir, f"{base}_document.jpg")
        cv2.imwrite(out_path, warped)

        result.update({
            "status": "success",
            "message": "Document extracted.",
            "coordinates": coords,
            "document_image": out_path,
            "homography": H.tolist(),
            "warp_size": {"width": warp_size[0], "height": warp_size[1]},
            "destination_corners": [[0.0, 0.0], [warp_size[0] - 1.0, 0.0], [warp_size[0] - 1.0, warp_size[1] - 1.0], [0.0, warp_size[1] - 1.0]],
        })
    except Exception as e:
        result["message"] = str(e)

    return result


def find_document_outline(img_path: str, output_dir: str = "data") -> Dict[str, Any]:
    """
    Run the DocTR outline detector and return the result JSON.

    Simplified public wrapper without raising on failure; errors are captured in
    the returned dictionary while preserving the standard output-path layout.

    Processing order:
        1. Run internal DocTR pipeline
        2. Return the result as-is (no exceptions raised)

    Args:
        img_path: Path to the input image.
        output_dir: Output directory or "data" for project data/.

    Returns:
        Dict[str, Any]: Same schema as `_find_document_doctr`.

    Raises:
        None.
    """
    return _find_document_doctr(img_path, output_dir=output_dir)

if __name__ == "__main__":
    TEST_IMAGES = [r"/Users/lutsch/Master-DEV/1. Semester/advanceddlteam5/data/layout_detector/images/test4.jpeg"]
    OUTPUT_BASE = OUTPUT_DIR
    
    # Load model once
    model = ocr_predictor(pretrained=True)

    for img_path in TEST_IMAGES:
        if not os.path.exists(img_path): continue
        sub_dir = os.path.join(OUTPUT_BASE, os.path.splitext(os.path.basename(img_path))[0])
        res = _find_document_doctr(img_path, output_dir=sub_dir, model=model)
        print(json.dumps(res, indent=2))
