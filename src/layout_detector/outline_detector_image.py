"""Document outline detection using Hough-lines-based pipeline.

Implements the flow described in https://medium.com/intelligentmachines/document-detection-in-python-2f9ffd26bf65:
resize → denoise → threshold → morph close → Canny → Hough lines → line intersections →
KMeans corners → perspective warp. Input/output schema matches the previous version.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points into a consistent TL, TR, BR, BL sequence.

    This helper normalizes arbitrary quad point orderings so downstream
    perspective transforms stay stable and predictable.

    Processing order:
        1. Compute coordinate sums to pick TL/BR
        2. Compute coordinate diffs to pick TR/BL
        3. Return ordered 4x2 array

    Args:
        pts: Array of shape (4, 2) containing four (x, y) corner points.

    Returns:
        np.ndarray: Ordered points shaped (4, 2) as TL, TR, BR, BL.

    Raises:
        None.
    """
    rect = np.zeros((4, 2), dtype="float32")
    # Sum of coordinates -> top-left has minimum, bottom-right has maximum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference -> top-right has min diff (x - y), bottom-left has max diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def resize_preserve(image: np.ndarray, max_dim: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Resize an image while preserving aspect ratio up to a maximum dimension.

    This keeps processing efficient by shrinking oversized inputs but returns
    the scale factor so detections can be mapped back to the original size.

    Processing order:
        1. Check if max(h, w) exceeds max_dim
        2. Compute scale and resize if needed
        3. Return resized image and scale factor

    Args:
        image: Input BGR image array.
        max_dim: Maximum allowed dimension (pixels) for height or width.

    Returns:
        Tuple[np.ndarray, float]: (resized_image, applied_scale).

    Raises:
        None.
    """
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, scale


def preprocess_for_edges(image: np.ndarray, output_dir: str, base: str) -> np.ndarray:
    """
    Prepare an image for Hough detection using an adaptive pipeline.

    Steps: CLAHE → denoise → blur → adaptive+Otsu threshold → close → percentile Canny.
    Debug artifacts are saved using the provided base.
    """
    save_debug_image("00_original_adapt", image, output_dir, base)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_debug_image("01_gray_adapt", gray, output_dir, base)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    save_debug_image("02_clahe_adapt", gray_eq, output_dir, base)

    denoised = cv2.fastNlMeansDenoising(gray_eq, h=10)
    save_debug_image("03_denoised_adapt", denoised, output_dir, base)

    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    save_debug_image("04_blurred_adapt", blurred, output_dir, base)

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,  # larger block to adapt better on colored backgrounds
        5,   # gentler bias to keep white pages bright
    )
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(adaptive, otsu)
    save_debug_image("05_binary_adapt", binary, output_dir, base)

    # Strong close to join gaps; 7x7 kernel with four passes.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=4)
    save_debug_image("06_closed_adapt", closed, output_dir, base)

    v = float(np.median(closed))
    lower = int(max(8.0, 0.45 * v))
    upper = int(min(230.0, 1.5 * v if v > 0 else 150.0))
    edges = cv2.Canny(closed, lower, upper)
    save_debug_image("07_edges_adapt", edges, output_dir, base)
    return edges

def hough_lines(edges: np.ndarray, threshold: int = 60) -> List[Tuple[float, float]]:
    """
    Detect straight lines in an edge map using the standard Hough transform.

    Processing order:
        1. Run cv2.HoughLines with fixed parameters
        2. Normalize output to a list of (rho, theta) pairs

    Args:
        edges: Binary edge map from Canny.

    Returns:
        List[Tuple[float, float]]: Detected lines as (rho, theta) pairs.

    Raises:
        None.
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=threshold)
    if lines is None:
        return []
    return [(float(rho), float(theta)) for [[rho, theta]] in lines]


def _angle_between(theta1: float, theta2: float) -> float:
    """
    Compute the acute angle between two line angles in degrees.

    Processing order:
        1. Take absolute difference
        2. Mirror to acute angle
        3. Convert radians to degrees

    Args:
        theta1: First angle in radians.
        theta2: Second angle in radians.

    Returns:
        float: Acute angle between inputs in degrees.

    Raises:
        None.
    """
    ang = abs(theta1 - theta2)
    ang = min(ang, np.pi - ang)
    return ang * 180.0 / np.pi


def _intersection(line1: Tuple[float, float], line2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """
    Compute the intersection point of two lines given in polar form.

    Processing order:
        1. Build the coefficient matrix for both lines
        2. Solve the 2x2 system if it is well-conditioned

    Args:
        line1: First line as (rho, theta).
        line2: Second line as (rho, theta).

    Returns:
        Optional[Tuple[float, float]]: (x, y) intersection or None if parallel.

    Raises:
        None.
    """
    (rho1, theta1), (rho2, theta2) = line1, line2
    mat = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)],
    ])
    if np.abs(np.linalg.det(mat)) < 1e-6:
        return None
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(mat, b)
    return float(x0[0]), float(y0[0])


def intersections_from_lines(lines: List[Tuple[float, float]], shape: Tuple[int, int], angle_tol: Tuple[float, float] = (70, 110)) -> List[Tuple[float, float]]:
    """
    Find pairwise intersections between near-orthogonal Hough lines inside image bounds.

    Processing order:
        1. Iterate unique line pairs and filter by angle tolerance
        2. Compute intersections
        3. Keep points within image extent

    Args:
        lines: List of Hough lines as (rho, theta).
        shape: Image shape as (height, width).
        angle_tol: Inclusive angle tolerance (deg) for near-right angles.

    Returns:
        List[Tuple[float, float]]: Valid intersection points within the image.

    Raises:
        None.
    """
    h, w = shape  # shape = (height, width)
    pts: List[Tuple[float, float]] = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            theta1, theta2 = lines[i][1], lines[j][1]
            ang = _angle_between(theta1, theta2)
            if ang < angle_tol[0] or ang > angle_tol[1]:
                continue
            inter = _intersection(lines[i], lines[j])
            if inter is None:
                continue
            x, y = inter
            if 0 <= x < w and 0 <= y < h:
                pts.append((x, y))
    return pts


def cluster_corners(points: List[Tuple[float, float]]) -> Optional[np.ndarray]:
    """
    Cluster intersection points into four corner centroids using KMeans.

    Processing order:
        1. Validate minimum point count
        2. Run KMeans with k=4
        3. Return cluster centers

    Args:
        points: Candidate intersection points.

    Returns:
        Optional[np.ndarray]: (4, 2) array of corner centers or None if insufficient data.

    Raises:
        None.
    """
    if len(points) < 4:
        return None
    data = np.array(points, dtype=np.float32)
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    kmeans.fit(data)
    return kmeans.cluster_centers_


def _clusters_compact_and_separated(points: List[Tuple[float, float]], centers: np.ndarray) -> bool:
    """Heuristic to ensure intersections form a plausible quad candidate."""
    if centers is None or centers.shape != (4, 2):
        return False
    pts = np.array(points, dtype=np.float32)
    if pts.size == 0:
        return False
    dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    nearest = dists.min(axis=1)
    # Relaxed compactness: allow up to ~80px spread on resized images.
    if float(nearest.max()) > 80.0:
        return False
    # Separation: allow closer centers but avoid collapse.
    center_dists = np.linalg.norm(centers[None, :, :] - centers[:, None, :], axis=2)
    min_center_dist = float(np.min(center_dists + np.eye(4) * 1e6))
    if min_center_dist < 20.0:
        return False
    return True

def transform_document(image: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]], np.ndarray, Tuple[int, int]]:
    """
    Warp a quadrilateral document region to a top-down view with sanity checks.

    Applies geometric validation, perspective transform, orientation correction,
    and blank-crop rejection to produce a usable document patch.

    Processing order:
        1. Order quad points and compute dimensions
        2. Validate area, angles, and side ratios
        3. Apply perspective warp
        4. Rotate to match portrait/landscape preference
        5. Reject nearly blank warps

    Args:
        image: Input BGR image containing the document.
        quad: (4, 2) array of document corner points.

    Returns:
        Tuple[np.ndarray, List[Dict[str, float]], np.ndarray, Tuple[int, int]]:
            - warped image
            - ordered corner metadata
            - homography matrix
            - warp size (width, height)

    Raises:
        ValueError: If validation fails (e.g., tiny area, bad angles, blank crop).
    """
    h_img, w_img = image.shape[:2]
    pts = quad.astype("float32")
    ordered = _order_points(pts).astype("float32")
    coordinate_labels = ["top_left", "top_right", "bottom_right", "bottom_left"]
    coordinates_struct = [
        {"label": lbl, "x": float(pt[0]), "y": float(pt[1])}
        for lbl, pt in zip(coordinate_labels, ordered.tolist())
    ]

    def _dist(a, b):
        return np.linalg.norm(a - b)

    (tl, tr, br, bl) = ordered
    widthA = _dist(br, bl)
    widthB = _dist(tr, tl)
    maxWidth = int(max(widthA, widthB))
    heightA = _dist(tr, br)
    heightB = _dist(tl, bl)
    maxHeight = int(max(heightA, heightB))
    if maxWidth < 10 or maxHeight < 10:
        raise ValueError("Detected outline too small to crop.")

    # Reject quads that are mostly outside the image or unreasonably small vs. image area.
    margin = 5.0
    if np.any(ordered[:, 0] < -margin) or np.any(ordered[:, 0] > (w_img + margin)) or np.any(ordered[:, 1] < -margin) or np.any(ordered[:, 1] > (h_img + margin)):
        raise ValueError("Detected outline is outside the image bounds.")

    poly_area = 0.5 * abs(
        tl[0] * tr[1] + tr[0] * br[1] + br[0] * bl[1] + bl[0] * tl[1]
        - tr[0] * tl[1] - br[0] * tr[1] - bl[0] * br[1] - tl[0] * bl[1]
    )
    img_area = float(h_img * w_img)
    if poly_area < 0.06 * img_area:
        raise ValueError("Detected outline area is too small relative to the image.")

    # Sanity checks on shape regularity
    bbox_area = maxWidth * maxHeight
    if bbox_area <= 0 or (poly_area / float(bbox_area)) < 0.65:
        raise ValueError("Detected outline is too skinny relative to its bounding box.")

    # Adjacent edge angles should be roughly orthogonal (within ~25 degrees of 90)
    def _cos_angle(v1, v2):
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 1.0
        return float(np.dot(v1, v2) / denom)

    edges = [tr - tl, br - tr, bl - br, tl - bl]
    cos_angles = [abs(_cos_angle(edges[i], edges[(i + 1) % 4])) for i in range(4)]
    if any(c > 0.5 for c in cos_angles):
        raise ValueError("Detected outline corners are not close to right angles.")

    # Opposite sides should be similar length to avoid extreme parallelograms
    def _ratio_ok(a, b, tol: float = 2.2):
        lo, hi = 1.0 / tol, tol
        rb = b if b != 0 else 1e-6
        r = a / rb
        return lo <= r <= hi

    if not _ratio_ok(widthA, widthB) or not _ratio_ok(heightA, heightB):
        raise ValueError("Detected outline side lengths are inconsistent.")

    # Require the quad to cover a reasonable span of the image in at least one dimension
    min_span = 0.16 * min(h_img, w_img)
    if maxWidth < min_span and maxHeight < min_span:
        raise ValueError("Detected outline spans too little of the image.")

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Decide target orientation from the warped size: keep landscape if width >= height.
    orientation = "landscape" if maxWidth >= maxHeight else "portrait"
    h_warp, w_warp = warped.shape[:2]
    if orientation == "portrait" and w_warp > h_warp:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == "landscape" and h_warp > w_warp:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Reject nearly blank warps (common when the quad is wrong and samples outside the page).
    if warped.std() < 2.0:
        raise ValueError("Detected outline produced an invalid (blank) crop.")

    return warped, coordinates_struct, M, (maxWidth, maxHeight)

def save_debug_image(name: str, img: np.ndarray, output_dir: str, base: str) -> None:
    """
    Save a debug image artifact to the layout-detector output directory.

    Processing order:
        1. Build file path with base and name
        2. Write image to disk

    Args:
        name: Suffix for the debug image filename.
        img: Image array to save.
        output_dir: Directory for saving.
        base: Base filename stem for grouping artifacts.

    Returns:
        None.

    Raises:
        None.
    """
    path = os.path.join(output_dir, f"{base}_{name}.jpg")
    cv2.imwrite(path, img)


def _fallback_doctr(img_path: str, output_dir: str, result: Dict[str, Any], message: str) -> Dict[str, Any]:
    """Fallback to OCR-guided DocTR detector when Hough is uncertain."""
    try:
        from .outline_detector_ocr import find_document_outline as find_doctr
    except Exception as e:  # pragma: no cover - defensive
        result["message"] = f"{message}; doctr import failed: {e}"
        return result

    doctr_res = find_doctr(img_path, output_dir=output_dir)
    doctr_res.setdefault("message", message)
    if doctr_res.get("status") != "success":
        doctr_res["message"] = f"{message}; doctr fallback also failed"
    return doctr_res

def find_document_outline(
    img_path: str,
    *,
    raise_on_error: bool = False,
    output_dir: str = "data",
) -> Dict[str, Any]:
    """
    Detect a document outline using Hough lines, intersections, and KMeans corners.

    This pipeline is geometry-driven and avoids OCR, producing a perspective-warped
    page crop plus metadata. Debug artifacts are saved alongside results.

    Processing order:
        1. Load image and validate path
        2. Preprocess for edges and run Hough lines
        3. Cluster intersections into corners
        4. Warp document with validation and orientation fix
        5. Save warped output and metadata

    Args:
        img_path: Absolute or relative path to the input image.
        raise_on_error: If True, raise on failure instead of returning an error dict.
        output_dir: Root directory for saving artifacts (subdir per image).

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
        FileNotFoundError: If the image cannot be loaded and raise_on_error is True.
        ValueError: For invalid inputs when raise_on_error is True.
    """
    result: Dict[str, Any] = {
        "status": "error",
        "message": "Uninitialized",
        "coordinates": None,
        "document_image": None,
        "homography": None,
        "warp_size": None,
        "destination_corners": None,
    }

    if not isinstance(img_path, str) or not img_path:
        result["message"] = "Image path must be a non-empty string."
        if raise_on_error:
            raise ValueError(result["message"])
        return result

    image = cv2.imread(img_path)
    if image is None:
        result["message"] = f"Failed to load image: {img_path}"
        if raise_on_error:
            raise FileNotFoundError(result["message"])
        return result

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_dir_img = os.path.join(output_dir, base)
    os.makedirs(out_dir_img, exist_ok=True)

    resized, scale = resize_preserve(image, max_dim=1100)

    base_tag = f"{base}_adapt"
    edges = preprocess_for_edges(resized, out_dir_img, base_tag)

    lines = hough_lines(edges)
    if not lines:
        return _fallback_doctr(img_path, output_dir, result, "adaptive: No Hough lines detected.")

    # Debug image for Hough lines visualization on the resized input
    hough_vis = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if resized.ndim == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    for rho, theta in lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(hough_vis, pt1, pt2, (0, 255, 0), 2)
    save_debug_image("hough_lines", hough_vis, out_dir_img, base_tag)

    inters = intersections_from_lines(lines, edges.shape[:2])
    if len(inters) < 4:
        return _fallback_doctr(img_path, output_dir, result, "adaptive: Not enough intersections for corners.")

    clusters = cluster_corners(inters)
    if clusters is None or clusters.shape[0] != 4:
        return _fallback_doctr(img_path, output_dir, result, "adaptive: Failed to cluster corners.")

    if not _clusters_compact_and_separated(inters, clusters):
        return _fallback_doctr(img_path, output_dir, result, "adaptive: Intersections not a clear quad candidate.")

    scale_back = 1.0 / scale
    quad = clusters * scale_back

    try:
        warped, coordinates_struct, H, warp_size = transform_document(image, quad)
    except ValueError as e:
        return _fallback_doctr(img_path, output_dir, result, f"adaptive: {e}")

    out_path = os.path.join(out_dir_img, f"{base}_document_hough_adapt.jpg")
    cv2.imwrite(out_path, warped)

    result.update({
        "status": "success",
        "message": "Document extracted via Hough lines (adaptive).",
        "coordinates": coordinates_struct,
        "document_image": out_path,
        "homography": H.tolist(),
        "warp_size": {"width": warp_size[0], "height": warp_size[1]},
        "destination_corners": [[0.0, 0.0], [warp_size[0] - 1.0, 0.0], [warp_size[0] - 1.0, warp_size[1] - 1.0], [0.0, warp_size[1] - 1.0]],
    })
    return result

# Previous separate annotation helper removed for simplicity.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hough-line document detector")
    parser.add_argument("--image", required=False, default="/Users/lutsch/Master-DEV/1. Semester/advanceddlteam5/data/layout_detector/images/test4.jpeg", help="Path to input image")
    parser.add_argument("--out", required=False, default="data", help="Output directory")
    args = parser.parse_args()

    res = find_document_outline(args.image, output_dir=args.out)
    print(json.dumps(res, indent=2))