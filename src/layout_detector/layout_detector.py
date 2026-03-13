r"""layout_detector.py
----------------------
High-level FastMCP-style layout detector packaged as a class.

This module exposes a `LayoutDetector` class whose methods wrap the document
outline detection pipeline and OCR, returning JSON results suitable for MCP
servers.

Provided interface:
- `LayoutDetector.get_outline(image_path, output_dir=None)`: Detect outline with
    fallback and return JSON.
- `LayoutDetector.get_ocr(image_path, output_dir=None)`: Run PaddleOCR and
    return artifact paths JSON.
- `LayoutDetector.get_outline_image(image_path, output_dir=None)`: Hough-lines
    detector only.
- `LayoutDetector.get_outline_ocr(image_path, output_dir=None)`: OCR-guided
    detector only.

Behavior:
- Writes artifacts to a caller-provided `output_dir`, or resolves one from the
    image path when omitted.
- Saves preprocessing/debug images to disk only (not included in JSON).
- Success JSON: includes `status`, `message`, outline metadata, and paths.
- Failure JSON: normalized with `status='error'`, `message`,
    `document_image=input_path`.

Integration:
- Instantiate `LayoutDetector` (optionally pass a custom output-dir resolver)
    and call its methods directly from MCP servers or other code.

Tools:
    LayoutDetector.get_outline: Run outline detection with fallback and return result JSON
    LayoutDetector.get_ocr: Run PaddleOCR and return artifact paths JSON
    LayoutDetector.get_outline_image: Run Hough-lines detector only
    LayoutDetector.get_outline_ocr: Run OCR-guided detector only

Example:
    from src.layout_detector.layout_detector import LayoutDetector
    detector = LayoutDetector()
    outline = detector.get_outline(r"C:\path\to\image.jpg")
    ocr = detector.get_ocr(r"C:\path\to\image.jpg")
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import os

from .outline_detector_image import find_document_outline as find_document_outline_image
from .outline_detector_ocr import find_document_outline as find_document_outline_ocr
from .paddle_ocr import run_ocr
from .utils import resolve_layout_output_dir


class LayoutDetector:
    """
    Layout detector wrapper with helpers for outline detection and OCR.

    Instantiate once (optionally with a custom output directory resolver) and
    call the methods to run detectors. If `output_dir` is omitted on calls,
    the resolver is used to build a stable path from `image_path`.

    Args:
        output_dir_resolver: Callable taking `image_path` and returning an
            output directory string. Defaults to `resolve_layout_output_dir`.
    """

    def __init__(
        self,
        output_dir_resolver: Callable[[str], str] = resolve_layout_output_dir,
    ) -> None:
        self._resolve_output_dir = output_dir_resolver

    def get_outline(
        self, image_path: str, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run outline detection with an automatic fallback and return JSON artifacts.

        This method first tries the fast Hough-lines detector, then falls back
        to the OCR-guided detector if needed. Artifacts are written to the
        resolved `output_dir` for downstream tools.

        Processing order:
            1. Run image-based detector (Hough-lines)
            2. If it fails, run OCR-guided detector
            3. Normalize failure output to include the input path

        Args:
            image_path: Absolute or relative path to the image to process.
            output_dir: Destination directory for artifacts. If omitted,
                the configured resolver is used.

        Returns:
            Dict[str, Any]: Result dictionary with keys:
                - status (str): "success" on extraction, otherwise "error"
                - message (str): Human-readable status or error message
                - coordinates (list|None): Corner metadata if available
                - document_image (str|None): Path to saved warped page (or input path on failure)
                - homography (list|None): 3x3 homography matrix serialized
                - warp_size (dict|None): Width/height of the warp
                - destination_corners (list|None): Target rectangle corners

        Raises:
            None explicitly; errors are captured in the returned dict.
        """
        target_dir = self._ensure_output_dir(image_path, output_dir)

        # First attempt: image-based detector (fast, geometry-driven)
        result = find_document_outline_image(image_path, output_dir=target_dir)

        # Fallback: OCR-guided detector if the first one failed
        if result.get("status") != "success":
            result = find_document_outline_ocr(image_path, output_dir=target_dir)

        # Ensure document_image path is always present; on failure, set to input path
        if result.get("status") != "success":
            result["document_image"] = image_path
        return result

    def get_ocr(
        self, image_path: str, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run PaddleOCR on an image and return saved artifact paths.

        This method standardizes OCR execution and output locations for the
        layout detector suite so that upstream tools can reliably find JSON
        and preview images under the resolved directory.

        Processing order:
            1. Ensure directory exists
            2. Run PaddleOCR and save outputs
            3. Normalize the result to always include input_path

        Args:
            image_path: Absolute or relative path to the image to OCR.
            output_dir: Destination directory for OCR artifacts. If omitted,
                the configured resolver is used.

        Returns:
            Dict[str, Any]: Result dictionary with keys:
                - status (str): "success" on OCR completion, otherwise "error"
                - input_path (str): Original image path
                - ocr_image (str|None): Path to rendered OCR visualization
                - ocr_json (str|None): Path to OCR JSON output
                - error (str|None): Error message if OCR failed

        Raises:
            None explicitly; errors are captured in the returned dict.
        """
        target_dir = self._ensure_output_dir(image_path, output_dir)

        result = run_ocr(image_path, save_dir=target_dir)
        if "input_path" not in result:
            result["input_path"] = image_path
        return result

    def get_outline_image(
        self, image_path: str, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run only the Hough-lines outline detector and return JSON artifacts.

        Use this when you want a geometry-only detector without OCR overhead.
        Outputs are written under the resolved layout detector directory.

        Processing order:
            1. Run Hough-lines detector
            2. Normalize failure output to include the input path

        Args:
            image_path: Absolute or relative path to the image to process.
            output_dir: Destination directory for artifacts. If omitted,
                the configured resolver is used.

        Returns:
            Dict[str, Any]: Result dictionary with keys:
                - status (str): "success" on extraction, otherwise "error"
                - message (str): Human-readable status or error message
                - coordinates (list|None): Corner metadata if available
                - document_image (str|None): Path to saved warped page (or input path on failure)
                - homography (list|None): 3x3 homography matrix serialized
                - warp_size (dict|None): Width/height of the warp
                - destination_corners (list|None): Target rectangle corners

        Raises:
            None explicitly; errors are captured in the returned dict.
        """
        target_dir = self._ensure_output_dir(image_path, output_dir)

        result = find_document_outline_image(image_path, output_dir=target_dir)
        if result.get("status") != "success":
            result["document_image"] = image_path
        return result

    def get_outline_ocr(
        self, image_path: str, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run only the OCR-guided outline detector and return JSON artifacts.

        Use this when you want text-aware detection that leverages OCR bounding
        boxes to infer the page boundary. Outputs follow the resolved layout
        detector directory scheme.

        Processing order:
            1. Run OCR-guided detector
            2. Normalize failure output to include the input path

        Args:
            image_path: Absolute or relative path to the image to process.
            output_dir: Destination directory for artifacts. If omitted,
                the configured resolver is used.

        Returns:
            Dict[str, Any]: Result dictionary with keys:
                - status (str): "success" on extraction, otherwise "error"
                - message (str): Human-readable status or error message
                - coordinates (list|None): Corner metadata if available
                - document_image (str|None): Path to saved warped page (or input path on failure)
                - homography (list|None): 3x3 homography matrix serialized
                - warp_size (dict|None): Width/height of the warp
                - destination_corners (list|None): Target rectangle corners

        Raises:
            None explicitly; errors are captured in the returned dict.
        """
        target_dir = self._ensure_output_dir(image_path, output_dir)

        result = find_document_outline_ocr(image_path, output_dir=target_dir)
        if result.get("status") != "success":
            result["document_image"] = image_path
        return result

    def _ensure_output_dir(self, image_path: str, output_dir: Optional[str]) -> str:
        """Resolve and create the output directory if needed."""
        target_dir = output_dir or self._resolve_output_dir(image_path)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir



