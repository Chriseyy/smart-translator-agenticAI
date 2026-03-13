"""
loader.py
---------
Image loading module for acquiring images from various sources.

This module provides the ImageLoader class which handles image acquisition
from both the file system and webcam, with support for interactive file
selection and camera capture.

Classes:
    ImageLoader: Main class for loading images from either a given path, 
    via file explorer or webcam capture.
"""

import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from .utils import ensure_data_folder, save_image


class ImageLoader:
    """
    Handles image acquisition from webcam or file system.
    
    This class provides methods for loading images from different sources:
    - File system (with optional GUI file picker)
    - Webcam capture with live preview
    
    Webcam captures are automatically saved to the data directory.
    File loads return the path without copying the file.
    
    Attributes:
        data_dir (str): Path to the data directory for saving captured images
        
    """
    
    def __init__(self):
        """Initialize the ImageLoader with data directory."""
        self.data_dir = ensure_data_folder()

    def load_image_from_path(self, path: str = None) -> dict:
        """
        Load an image from a file path or via interactive file dialog.
        
        If no path is provided, opens a GUI file picker dialog for the user
        to select an image file. Supports common image formats (PNG, JPG,
        JPEG, BMP, TIFF).
        
        Note: This method only validates and returns the path. The image file
        is not copied or modified.
        
        Args:
            path: Absolute or relative path to an image file. If None,
                opens a file dialog for interactive selection.
                
        Returns:
            dict: Result dictionary with keys:
                - status (str): "success" if image was found/selected
                - path (str): Absolute path to the image file
                
        Raises:
            FileNotFoundError: If path is None and no file is selected in dialog,
                or if the specified path does not exist
        """
        # If no path provided, open file dialog for interactive selection
        if path is None:
            root = Tk()
            try:
                root.withdraw()
                root.attributes('-topmost', True)
                root.update()
                path = askopenfilename(
                    title="Select an image file",
                    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")],
                    parent=root,
                )
            finally:
                root.destroy()

            if not path:
                raise FileNotFoundError("No file selected in dialog.")

        # Validate that the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        # Return absolute path for consistency
        return {"status": "success", "path": os.path.abspath(path)}

    def capture_from_webcam(self, camera_index: int = 0) -> dict:
        """
        Capture an image from the webcam with live preview.
        
        This method opens a live camera preview window where the user can:
        - Press SPACE to capture the current frame
        - Press ESC to cancel the operation
        
        The captured image is automatically saved to the data directory
        with a timestamped filename.
        
        Args:
            camera_index: Index of the camera to use (default: 0 for primary camera).
                Multiple cameras can be accessed with different indices (1, 2, etc.)
                
        Returns:
            dict: Result dictionary with keys:
                - status (str): "success" if image was captured, "cancelled" if user cancelled
                - path (str): Absolute path to saved image (only present if status is "success")
                - message (str): Error message when status is "error"
        """
        # Initialize video capture with specified camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {
                "status": "error",
                "message": f"Could not open webcam at index {camera_index}. Check connection/permissions.",
            }

        print("Camera open - Press SPACE to capture, ESC to cancel.")
        frame = None
        
        # Live preview loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display live preview
            cv2.imshow("Camera - Press SPACE to capture", frame)
            
            # Check for user input
            key = cv2.waitKey(1) & 0xFF  # Mask to get only the lower 8 bits
            
            if key == 27:  # ESC key
                print("Capture cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                return {"status": "cancelled"}
                
            elif key == 32:  # SPACE key
                print("Image captured!")
                break

        # Clean up camera and windows
        cap.release()
        cv2.destroyAllWindows()

        # Validate that we have a frame
        if frame is None:
            return {"status": "error", "message": "No frame captured from webcam."}

        # Save the captured frame using utility function
        filepath = save_image(frame, self.data_dir, prefix="capture")
        
        return {"status": "success", "path": filepath}


