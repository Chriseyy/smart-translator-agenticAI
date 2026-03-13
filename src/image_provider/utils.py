"""
utils.py
--------
Utility functions for file management and image saving operations
for the Image Provider component.

This module provides reusable helper functions for:
- Creating and managing the data directory
- Saving images in various formats (numpy arrays, PIL Images)
"""

import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image


def ensure_data_folder() -> str:
    """
    Ensure that a 'data' directory exists in the current working directory. 
    As the image provider is the first step of the translation pipeline, this needs to make sure
    the data directory for saving images is present.
    
    This function creates the data folder if it doesn't exist and is idempotent,
    meaning it can be safely called multiple times without side effects.
    
    Returns:
        str: Absolute path to the data directory
        
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def save_image(img, data_dir: str, prefix: str = "image") -> str:
    """
    Save an image to the specified data directory with a timestamped filename.
    
    This function handles both OpenCV numpy arrays (BGR format) and PIL Images,
    automatically detecting the type and using the appropriate save method.
    
    Args:
        img: Image to save. Can be either:
            - numpy.ndarray: OpenCV image in BGR format
            - PIL.Image.Image: PIL Image object
        data_dir: Directory path where the image should be saved
        prefix: Prefix for the filename (default: "image")
    
    Returns:
        str: Absolute path to the saved image file
        
    Raises:
        ValueError: If img is neither a numpy array nor a PIL Image
    """
    # Generate timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(data_dir, filename)
    
    # Handle different image types
    if isinstance(img, np.ndarray):
        # OpenCV numpy array - save directly with cv2
        cv2.imwrite(filepath, img)
    elif isinstance(img, Image.Image):
        # PIL Image - convert to RGB and save with high quality
        img.convert("RGB").save(filepath, format="JPEG", quality=95)
    else:
        raise ValueError(
            f"Unsupported image type: {type(img)}. "
            "Expected numpy.ndarray or PIL.Image.Image"
        )
    
    return filepath