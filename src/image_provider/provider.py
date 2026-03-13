"""
provider.py
-----------
ImageProvider class combining image loading and preprocessing utilities.

This module exposes an `ImageProvider` class that wraps the loader and
preprocessor functionality into a single, MCP-friendly interface:
- Capturing images from webcam
- Loading images from file paths
- Unified image provisioning interface
- Image preprocessing (automatic or manual)

Instantiate `ImageProvider` once and use its methods directly or register
them as tools in the server layer.

Tools:
    ImageProvider.load_image: Unified interface for webcam or file sources (no preprocessing)
    ImageProvider.capture_from_webcam: Capture image from webcam
    ImageProvider.load_image_from_path: Load image from a file path/dialog
    ImageProvider.auto_preprocess_image: Preprocess from an existing file with fixed defaults
    ImageProvider.preprocess_image: Preprocess from an existing file with manual settings
"""
from .loader import ImageLoader
from .preprocessor import ImagePreprocessor


class ImageProvider:
    """
    Unified image loading and preprocessing interface.

    Wraps ImageLoader and ImagePreprocessor so servers can depend on a single
    class for all image acquisition and enhancement operations.
    """

    def __init__(self) -> None:
        self._loader = ImageLoader()
        self._preprocessor = ImagePreprocessor()

    def load_image(self, source: str = "webcam", path: str | None = None) -> dict:
        """
        Unified interface for acquiring images from webcam or file without preprocessing.

        Args:
            source: "webcam" or "path". If "path", uses `path` or opens a dialog.
            path: File path to image (required when source="path" without dialog).

        Returns:
            dict: Result dictionary with status/path depending on source.

        Raises:
            ValueError: If source is not "webcam" or "path".
            FileNotFoundError: For missing file when source="path".
        """
        if source == "path":
            return self._loader.load_image_from_path(path)
        if source == "webcam":
            return self._loader.capture_from_webcam()
        raise ValueError(f"Invalid source '{source}'. Must be 'webcam' or 'path'")

    def load_image_from_path(self, path: str | None = None) -> dict:
        """Load an image via path or file dialog (delegates to ImageLoader)."""
        return self._loader.load_image_from_path(path)

    def capture_from_webcam(self, camera_index: int = 0) -> dict:
        """Capture an image from webcam (delegates to ImageLoader)."""
        return self._loader.capture_from_webcam(camera_index)

    def auto_preprocess_image(self, path: str) -> dict:
        """Run fixed automatic preprocessing on an existing image file."""
        return self._preprocessor.auto_process(path)

    def preprocess_image(
        self,
        path: str,
        enhance_contrast: float | None = None,
        brightness: float | None = None,
        denoise_strength: float | None = None,
        sharpen_strength: float | None = None,
    ) -> dict:
        """Process and enhance an image with manual parameters."""
        return self._preprocessor.process(
            path=path,
            enhance_contrast=enhance_contrast,
            brightness=brightness,
            denoise_strength=denoise_strength,
            sharpen_strength=sharpen_strength,
        )


