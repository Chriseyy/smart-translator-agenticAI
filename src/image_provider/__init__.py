"""
Image Provider Package
======================

Image acquisition and preprocessing utilities for document images.

This package provides:
- Loading images from file system (with GUI file picker)
- Capturing images from webcam with live preview
- Automatic and manual image preprocessing
- A unified `ImageProvider` class for MCP/tool integrations

Modules:
    utils: Utility functions for file management and image saving
    loader: Image acquisition from webcam and file system
    preprocessor: Image preprocessing and enhancement
    provider: `ImageProvider` class combining loader and preprocessor

"""

__version__ = "1.0.0"
__author__ = "Stefan Lutsch"

from .provider import ImageProvider

__all__ = ["ImageProvider"]



