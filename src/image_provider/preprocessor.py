"""
preprocessor.py
---------------
Image preprocessing module for document image enhancement.

This module provides both automatic and manual preprocessing capabilities
for document images using the Pillow (PIL) library. It supports various
enhancement operations including contrast adjustment, brightness control,
sharpening and denoising.

Classes:
    ImagePreprocessor: Main class for image preprocessing operations
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from .utils import ensure_data_folder, save_image


class ImagePreprocessor:
    """
    Handles automatic and manual preprocessing for document images.
    
    This class provides methods for enhancing document images through various
    preprocessing operations. It can work with both PIL Images and OpenCV
    numpy arrays.
    
    Attributes:
        data_dir (str): Path to the data directory where processed images are saved
    """
    
    def __init__(self):
        """Initialize the ImagePreprocessor with data directory."""
        self.data_dir = ensure_data_folder()

    def auto_process(self, path: str) -> dict:
        """
        Perform lightweight automatic preprocessing for document clarity.
        Uses fixed gentle defaults shared with the manual pipeline helpers.
        Args:
            path: Absolute or relative path to the image file to process
        Returns:
        dict: Result dictionary with keys:
            - status (str): "success" if processing completed
            - path (str): Absolute path to the saved auto processed image
        Fixed Args:
            - Contrast factor: 1.1
            - Brightness factor: 1.2
            - Denoise strength: None (no denoising)
            - Sharpen strength: 1.0
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        with Image.open(path) as img:
            img = self._to_rgb_image(img)
            img = self._apply_contrast(img, factor=1.1)
            img = self._apply_brightness(img, factor=1.2)
            img = self._apply_denoise(img, denoise_strength=None)
            img = self._apply_sharpen(img, sharpen_strength=1.0)

            original_basename = os.path.splitext(os.path.basename(path))[0]
            prefix = f"auto_preprocessed_{original_basename}"
            out_path = save_image(img, self.data_dir, prefix=prefix)

        return {"status": "success", "path": out_path}

    def process(
        self,
        path: str,
        enhance_contrast: float | None = None,
        brightness: float | None = None,
        denoise_strength: float | None = None,
        sharpen_strength: float | None = None,
    ) -> dict:
        """
        Apply user-selected preprocessing operations on an image file.
        
        This method provides fine-grained control over image preprocessing,
        allowing users to selectively apply various enhancement operations.
        Operations are applied in an optimized order to maximize quality.
        
        Processing order:
            1. Contrast enhancement (if specified)
            2. Brightness adjustment (if specified)
            3. Denoising (if specified)
            4. Sharpening (if specified)

        Args:
            path: Absolute or relative path to the image file to process
            enhance_contrast: Contrast multiplier (1.0 = original, <1.0 = less contrast,
                >1.0 = more contrast). None means no adjustment. Typical values: 1.1-1.5
            brightness: Brightness multiplier (1.0 = original, <1.0 = darker,
                >1.0 = brighter). None means no adjustment
            denoise_strength: Strength of denoising. None means no denoising. Typical
                range: 0.0–1.0. Values > 0.0 increase the effect.
            sharpen_strength: Strength of sharpening. None means no sharpening. Typical
                range: 0.0–2.0. Values > 0.0 increase the effect.

        Returns:
            dict: Result dictionary with keys:
                - status (str): "success" if processing completed
                - path (str): Absolute path to the saved processed image

        Raises:
            FileNotFoundError: If the specified image path does not exist
            
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        with Image.open(path) as img:
            img = self._to_rgb_image(img)

            
            img = self._apply_contrast(img, factor=enhance_contrast)
            img = self._apply_brightness(img, factor=brightness)
            img = self._apply_denoise(img, denoise_strength=denoise_strength)
            img = self._apply_sharpen(img, sharpen_strength=sharpen_strength)

            original_basename = os.path.splitext(os.path.basename(path))[0]
            prefix = f"preprocessed_{original_basename}"
            out_path = save_image(img, self.data_dir, prefix=prefix)

        return {"status": "success", "path": out_path}

    # --- Internal helpers ---
    def _to_rgb_image(self, image) -> Image.Image:
        """Convert PIL or OpenCV BGR array to RGB PIL image."""
        if isinstance(image, np.ndarray):
            import cv2

            rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_array).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError(
            f"Unsupported image type: {type(image)}. Expected numpy.ndarray or PIL.Image.Image"
        )

    def _apply_contrast(self, img: Image.Image, factor: float | None) -> Image.Image:
        """Apply contrast scaling when a factor is provided."""
        if factor is None:
            return img
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def _apply_brightness(self, img: Image.Image, factor: float | None) -> Image.Image:
        """Apply brightness scaling when a factor is provided."""
        if factor is None:
            return img
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def _apply_denoise(self, img: Image.Image, denoise_strength: float | None) -> Image.Image:
        """Median filter denoise mapped from a 0..1 strength."""
        if denoise_strength is None or denoise_strength <= 0:
            return img
        size = max(3, int(denoise_strength * 5) | 1)  # odd kernel >=3
        return img.filter(ImageFilter.MedianFilter(size=size))

    def _apply_sharpen(self, img: Image.Image, sharpen_strength: float | None) -> Image.Image:
        """Unsharp mask sharpening scaled by strength."""
        if sharpen_strength is None or sharpen_strength <= 0:
            return img
        percent = int(150 * sharpen_strength)
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))


