"""Document layout tools.

Overview
- Outline detection: extract a document region from an input image and warp it
  to a top-down view (current default implementation is DocTR-based).
- OCR: run PaddleOCR and save model-native artifacts (image + JSON) via
  `paddle_ocr.run_ocr`, with small shims to ensure compatibility.

Modules
- `outline_detector_image`: Hough-lines outline detector (geometry-driven).
- `outline_detector_ocr`: OCR-guided outline detector (text-aware fallback).
- `layout_detector`: Class wrapper (`LayoutDetector`) exposing both detectors
  plus OCR in a unified API.
- `paddle_ocr`: OCR runner + CLI returning artifact paths.
- `utils`: compatibility helpers applied before importing `paddleocr`.

"""

__version__ = "1.0.0"
__author__ = "Stefan Lutsch"

from .layout_detector import LayoutDetector

__all__ = ["LayoutDetector"]