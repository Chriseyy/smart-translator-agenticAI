import datetime
import json
import os
from typing import List, Dict, Tuple, Union 

import cv2 
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from simple_lama_inpainting import SimpleLama

class ImageRenderer:
    """
    Manages the 4-step pipeline for rendering translated text onto a document image.

    This class handles:
    1. Removal of original text using AI inpainting (SimpleLama).
    2. Drawing translated text with auto-fitting and texture matching.
    3. Applying inverse perspective transformations (Warping) with sharpness optimizations.
    4. Merging the result back onto the original image using seamless alpha blending.

    Attributes
    ----------
    device : str
        The computation device ('cpu' or 'cuda').
    default_font_name : str
        The name of the font file to use as a global fallback (e.g., 'arial.ttf').
    default_font_size : int
        The initial font size for text fitting.
    debug : bool
        If True, prints detailed logs to console.
    inpainting_model : SimpleLama
        The loaded AI model instance for text removal.
    """
    
    def __init__(self, 
                 device: str = "cpu", 
                 default_font_name: str = "arial.ttf",  # global fallback
                 default_font_size: float = 12.0,       # global fallback
                 debug: bool = False):
        """
        Initializes the ImageRenderer, loads the SimpleLama model and sets defaults.
        """
        self.debug = debug 
        self.default_font_name = default_font_name
        self.default_font_size = int(default_font_size)
        
        if self.debug: print("Initializing SimpleLama inpainting model...")
        try:
            # Auto-detect CUDA availability if requested
            self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            # Initialize model (downloads automatically if needed)
            self.inpainting_model = SimpleLama(device=self.device)
            if self.debug: print(f"  ImageRenderer (SimpleLama) initialized on {self.device}.")
        except Exception as e:
            if self.debug: print(f"  FAILED to initialize SimpleLama. Error: {e}")
            raise e
    
    # --- HELPER: Static Utilities ---
    @staticmethod
    def _get_distance(p1: Union[List, np.ndarray], p2: Union[List, np.ndarray]) -> float:
        """Calculate the Euclidean distance between two points."""
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    @staticmethod
    def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Converts a PIL Image (RGB) to an OpenCV NumPy array (BGR)."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Converts an OpenCV NumPy array (BGR) to a PIL Image (RGB)."""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    # --- HELPER: Font Loading ---
    def _load_font(self, font_identifier: str, font_size: int) -> ImageFont.FreeTypeFont:
        """
        Loads a TrueType font. 
        Strategy: Checks absolute path -> Checks local 'assets/fonts' -> Fallback to default.
        """
        # Check if identifier is a valid direct path
        if os.path.exists(font_identifier):
            try: 
                return ImageFont.truetype(font_identifier, font_size)
            except: 
                pass 

        # Construct path relative to project root's 'assets/fonts'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Traverse up to find project root (assuming standard structure)
        project_root = os.path.dirname(os.path.dirname(current_dir)) 
        if "assets" not in os.listdir(project_root):
             project_root = os.path.dirname(current_dir)

        fonts_dir = os.path.join(project_root, "assets", "fonts")
        
        clean_name = os.path.basename(font_identifier).lower().strip()
        filename = clean_name
        if not filename.endswith(".ttf"):
            filename += ".ttf"
        target_path = os.path.join(fonts_dir, filename)

        # Try loading target or fallback
        try:
            if os.path.exists(target_path): 
                return ImageFont.truetype(target_path, font_size)
            
            fallback_path = os.path.join(fonts_dir, self.default_font_name)
            if os.path.exists(fallback_path): 
                return ImageFont.truetype(fallback_path, font_size)
            
            return ImageFont.load_default()
        except Exception as e:
            if self.debug: print(f"    [Error] Failed to load font {font_identifier}: {e}")
            return ImageFont.load_default()
    
    # --- HELPER: Data Parsing ---
    def _parse_ocr_json(self, json_path: str) -> List[Dict]:
        """
        Reads the OCR JSON output. Extracts text, boxes, and specific fonts per box.
        
        Expected JSON format:
        {
            "rec_texts": ["Text A", ...],
            "rec_boxes": [[x,y,w,h], ...],
            "rec_font_names": ["/path/Arial.ttf", ...],
            "rec_font_sizes": [12.5, ...]
        }

        Returns
        -------
        List[Dict]
            List of dicts with keys: 'text', 'box', 'font_name', 'font_size'.
        """
        if not os.path.exists(json_path): 
            raise FileNotFoundError(f"OCR JSON not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
            
        rec_texts = data.get("rec_texts", [])
        rec_boxes = data.get("rec_boxes", [])
        rec_font_names = data.get("rec_font_names", [])
        rec_font_sizes = data.get("rec_font_sizes", [])

        parsed_boxes = []
        limit = min(len(rec_texts), len(rec_boxes))
        
        for i in range(limit):
            box = rec_boxes[i]
            if len(box) == 4:
                item = {
                    "text": rec_texts[i],
                    "box": [int(c) for c in box]
                }
                
                # Inject font info if available
                if i < len(rec_font_names) and rec_font_names[i]:
                    item["font_name"] = rec_font_names[i]
                if i < len(rec_font_sizes) and rec_font_sizes[i]:
                    try:
                        item["font_size"] = float(rec_font_sizes[i])
                    except (ValueError, TypeError):
                        pass
                parsed_boxes.append(item)
                        
        if self.debug: print(f"    [Data] Parsed {len(parsed_boxes)} text boxes from JSON.")
        return parsed_boxes

    # --- HELPER: Calculation ---
    def _calculate_transform_data(self, 
                                  layout_coords: Union[str, List[Dict]], 
                                  cropped_image_path: str, 
                                  original_image_path: str) -> Dict:
        """
        Calculates the Homography Matrix to map points from the cropped image 
        back to the original perspective image. Handles basic auto-rotation checks.
        """
        # Determine Output Shape (Size of the Original Image)
        if os.path.exists(original_image_path):
            with Image.open(original_image_path) as img:
                orig_w, orig_h = img.size
        else:
            orig_w, orig_h = 1920, 1080 # Fallback
            
        # Load Coordinates
        coords = []
        if isinstance(layout_coords, str):
            # Case A: Input is a file path -> Load JSON
            if not os.path.exists(layout_coords):
                if self.debug: print(f"    [Warning] Layout JSON file not found: {layout_coords}")
                return {"matrix": np.eye(3).tolist(), "output_shape": [orig_w, orig_h]}
            with open(layout_coords, 'r') as f:
                data = json.load(f)
            coords = data.get("coordinates", [])
        elif isinstance(layout_coords, list):
            # Case B: Input is already a list of coordinates
            coords = layout_coords

        else:
            if self.debug: print(f"    [Error] Invalid input for layout data: {type(layout_coords)}")
            return {"matrix": np.eye(3).tolist(), "output_shape": [orig_w, orig_h]}

        # Calculate Perspective Transform
        points_map = {pt['label']: [pt['x'], pt['y']] for pt in coords if 'label' in pt}
        required = ["top_left", "top_right", "bottom_right", "bottom_left"]
        
        if all(key in points_map for key in required):
            # Destination: Coordinates in the Original Image
            dst_points = np.float32([
                points_map["top_left"], points_map["top_right"],
                points_map["bottom_right"], points_map["bottom_left"]
            ])
            
            # Source: Corners of the Cropped Image
            with Image.open(cropped_image_path) as img:
                crop_w, crop_h = img.size
            
            # Standard Mapping (Top-Left to Top-Left, etc.)
            src_points = np.float32([
                [0, 0], [crop_w, 0], 
                [crop_w, crop_h], [0, crop_h]
            ])
            
            dst_w_px = self._get_distance(points_map["top_left"], points_map["top_right"])
            dst_h_px = self._get_distance(points_map["top_left"], points_map["bottom_left"])
            
            # Check: Is the target area Landscape, but the Crop is Portrait?
            # This happens if the layout detector rotated the crop to make text upright.
            if (dst_w_px > dst_h_px) and (crop_w < crop_h):
                if self.debug: print("    [Info] Rotation detected (Landscape Target -> Portrait Crop). Adjusting mapping.")
                
                # Remap src points 90 degrees clockwise to match destination
                # Original Top-Left corresponds to Crop Bottom-Left
                # Original Top-Right corresponds to Crop Top-Left
                # ...
                src_points = np.float32([
                    [0, crop_h],      # TL -> BL
                    [0, 0],           # TR -> TL
                    [crop_w, 0],      # BR -> TR
                    [crop_w, crop_h]  # BL -> BR
                ])
            
            # Calculate Homography: src -> dst
            M = cv2.getPerspectiveTransform(src_points, dst_points) 
            return {"matrix": M.tolist(), "output_shape": [orig_w, orig_h]}
        
        else:
            if self.debug: print("    [Error] Missing corner labels in layout data. Using Identity.")
            return {"matrix": np.eye(3).tolist(), "output_shape": [orig_w, orig_h]}
        
    # --- HELPER: Text Fitting ---
    def _fit_text(self, 
                  draw: ImageDraw.ImageDraw, 
                  text: str, 
                  box_width: int, 
                  box_height: int, 
                  font_name: str, 
                  max_font_size: int) -> Tuple[ImageFont.FreeTypeFont, str]:
        """
        Fits text into a bounding box by iteratively reducing font size and word-wrapping.
        """
        min_font_size = 6
        current_size = max_font_size
        
        while current_size >= min_font_size:
            font = self._load_font(font_name, int(current_size))
            words = text.split()
            lines = []
            current_line = []
            
            # Word Wrapping Logic
            for word in words:
                test_line = ' '.join(current_line + [word])
                # Check width in pixels
                if font.getlength(test_line) <= box_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        # Word is wider than box, force break
                        lines.append(word)
                        current_line = []
                        
            if current_line: lines.append(' '.join(current_line))

            final_text = '\n'.join(lines)
            
            # Check Total Height
            bbox = draw.multiline_textbbox((0, 0), final_text, font=font, spacing=0)
            if (bbox[3] - bbox[1]) <= box_height:
                return font, final_text # It fits, JUHU!
            
            current_size -= 1 # Reduce size and retry
            
        if self.debug: print(f"    [Warning] Text could not fit in box even at size {min_font_size}.")
        return self._load_font(font_name, min_font_size), text

    # --- HELPER: Texture simulation (against ‘spots’) ---
    def _add_simulated_noise(self, 
                             image: Image.Image, 
                             mask: Image.Image, 
                             intensity: int = 15) -> Image.Image:
        """
        Adds slight Gaussian noise (grain) to the masked areas.
        
        Purpose: 
        Inpainting models produce very smooth textures. This function adds grain 
        back to make the inpainted area match the surrounding paper texture.
        """
        if self.debug: print("    [Texture] Adding paper grain to inpainted areas...")
        
        # Fix dimension mismatch (SimpleLama often pads dimensions to multiples of 8)
        if image.size != mask.size:
            if self.debug: print(f"    [Warning] Resizing inpainted image from {image.size} to {mask.size} to match mask.")
            image = image.resize(mask.size, Image.Resampling.LANCZOS)

        cv_img = self._pil_to_cv2(image)
        cv_mask = np.array(mask) 

        # Generate Gaussian Noise
        noise = np.zeros(cv_img.shape, dtype=np.uint8)
        cv2.randn(noise, (0,0,0), (intensity, intensity, intensity)) # Intensity controls the strength
        
        # Add noise to the image (with saturation)
        noisy_img = cv2.add(cv_img, noise)
               
        # Apply only to masked (inpainted) areas
        mask_bool = cv_mask > 128
        final_img = cv_img.copy()
        final_img[mask_bool] = noisy_img[mask_bool]

        return self._cv2_to_pil(final_img)

    # --- PIPELINE STEPS ---
    def _inpaint_text_areas(self, image_path: str, boxes: List[Dict]) -> Image.Image:
        """
        Step 1: Removes original text using AI Inpainting.
        Includes masking, noise simulation, and mild sharpening.
        """
        if self.debug: print(f"    [Step 1] Removing text (SimpleLama)...")

        image = Image.open(image_path).convert("RGB")
        mask = Image.new("L", image.size, 0) # 0 = black background
        draw = ImageDraw.Draw(mask)

        for item in boxes:
            box = item['box']
            # Minimal 'bloat' (+2px) to ensure edges of text are fully covered
            draw.rectangle([box[0]-2, box[1]-2, box[2]+2, box[3]+2], fill=255)
        
        # AI Inference
        inpainted = self.inpainting_model(image, mask)

        # Post-Processing: Grain & Sharpening
        # inpainted_with_grain = self._add_simulated_noise(inpainted, mask, intensity=12)
        
        if self.debug: print("    [Step 1b] Applying mild sharpening...")
        sharpened = inpainted.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))

        return sharpened

    def _draw_translated_text(self, inpainted_image: Image.Image, text_boxes: List[Dict]) -> Image.Image:
        """
        Step 2: Draws the translated text onto the clean image.
        """
        if self.debug: print("    [Step 2] Drawing translated text...")

        img_copy = inpainted_image.copy()
        draw = ImageDraw.Draw(img_copy)

        for item in text_boxes:
            text = item.get('text', '')
            box = item['box']
            
            # Add padding to avoid overlapping lines
            padding_x, padding_y = 0.25, 0.25 # optional: distance from text to top or bottom box-border 

            box_w = (box[2] - box[0]) - (padding_x * 2)
            box_h = (box[3] - box[1]) - (padding_y * 2) # optional: Increases the distance between the top or bottom edge of the box and the text. (Idea: Text is not overlapping) 
            
            # Safety check for negative dimensions
            if box_w <= 0: box_w = box[2] - box[0]; padding_x = 0
            if box_h <= 0: box_h = box[3] - box[1]; padding_y = 0

            f_name = item.get("font_name", self.default_font_name)
            f_size = item.get("font_size", self.default_font_size)
            
            # Fit & Draw
            font, wrapped_text = self._fit_text(draw, text, box_w, box_h, f_name, int(f_size))
            
            draw.multiline_text((box[0] + padding_x, box[1] + padding_y), wrapped_text, font=font, fill=(0, 0, 0), spacing=0) 

        return img_copy

    def _apply_inverse_transform(self, rendered_text_image: Image.Image, transform_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 3: Warps the flat image back to perspective.
        Features High-Quality Lanczos interpolation and Post-Warp Sharpening.
        """
        if self.debug: print("    [Step 3] Warping image...")
        
        matrix = np.array(transform_data['matrix'])
        out_w, out_h = transform_data['output_shape']
        
        cv_img = self._pil_to_cv2(rendered_text_image)

        # Warp Image
        # INTER_LANCZOS4: High quality resampling (prevents blur).
        # BORDER_REPLICATE: Extends edge colors (avoids black/white border artifacts).
        warped_img = cv2.warpPerspective(
            cv_img, 
            matrix, 
            (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE 
            )

        # Post-Warp Sharpening (Unsharp Masking via OpenCV)
        # Formula: sharpened = original * 1.5 - blur * 0.5
        gaussian_blur = cv2.GaussianBlur(warped_img, (0, 0), 3.0)
        warped_img = cv2.addWeighted(warped_img, 1.5, gaussian_blur, -0.5, 0)
        
        # Warp Mask
        h_crop, w_crop = cv_img.shape[:2]
        white_crop = np.ones((h_crop, w_crop), dtype=np.uint8) * 255

        warped_mask = cv2.warpPerspective(
            white_crop, 
            matrix, 
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
            )
        
        # Seamless Blending Prep (Erosion & Blur)
        kernel = np.ones((3, 3), np.uint8)
        warped_mask = cv2.erode(warped_mask, kernel, iterations=1)
        warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 0)

        return warped_img, warped_mask

    def _add_result_to_image(self, warped_img: np.ndarray, mask: np.ndarray, original_image_path: str) -> Image.Image:
        """
        Step 4: Merges the warped document onto the original background using Alpha Blending.
        """
        if self.debug: print("    [Step 4] Merging with background...")
        
        original_pil = Image.open(original_image_path).convert("RGB")
        original_cv = self._pil_to_cv2(original_pil)

        # Safety: Resize original if dimensions mismatch (rare edge case)
        if original_cv.shape[:2] != warped_img.shape[:2]:
            if self.debug: print(f"    [Warning] Resizing original {original_cv.shape} to match output {warped_img.shape}")
            original_cv = cv2.resize(original_cv, (warped_img.shape[1], warped_img.shape[0]))

        
        # Normalize mask to 0.0 - 1.0 range for alpha blending
        alpha = mask.astype(float) / 255.0
        # Expand to 3 channels for RGB multiplication
        alpha = cv2.merge([alpha, alpha, alpha])
        
        # Alpha Blending Formula: 
        # Output = (Foreground * Alpha) + (Background * (1 - Alpha))
        foreground = warped_img.astype(float)
        background = original_cv.astype(float)
        
        combined = cv2.multiply(alpha, foreground) + cv2.multiply(1.0 - alpha, background)
        
        return self._cv2_to_pil(combined.astype(np.uint8))

    # --- MAIN PIPELINE ---
    def render_translated_image(self, 
                                cropped_image_path: str, 
                                original_image_path: str, 
                                ocr_json_path: str, 
                                layout_coords: Union[str, List[Dict]]) -> Dict:
        """
        Executes the full pipeline:
        1. Parse Data -> 2. Inpaint -> 3. Draw Text -> 4. Warp -> 5. Merge -> 6. Save
        """
        if self.debug: print(f"\n--- Rendering Pipeline Start ---")
        
        try:     
            try:
                # Data Preparation
                if self.debug: print(f"    [Data] Loading text boxes from {os.path.basename(ocr_json_path)}...")
                text_boxes = self._parse_ocr_json(ocr_json_path)
                
                if self.debug: print(f"    [Data] Calculating matrix...")
                transform_data = self._calculate_transform_data(layout_coords, cropped_image_path, original_image_path)
                
                if self.debug: print(f"    -> Data preparation complete.")
            except Exception as e:
                if self.debug: print(f"[Error] Data preparation failed: {e}")
                return {"status": "error", "message": f"Data Preparation Error: {e}"}         
    
            # Pipeline Execution
            # Step 1: Remove text
            try:
                img_step1 = self._inpaint_text_areas(cropped_image_path, text_boxes)
            except Exception as e:
                return {"status": "error", "message": f"Inpainting Error: {e}"}
            
            # Step 2: Draw new text (USES PER-BOX FONTS)
            try:
                img_step2 = self._draw_translated_text(img_step1, text_boxes)
            except Exception as e:
                return {"status": "error", "message": f"Text Rendering Error: {e}"}
            
            # Step 3: Warp back
            try:
                warped_img, mask = self._apply_inverse_transform(img_step2, transform_data)
            except Exception as e:
                return {"status": "error", "message": f"Warping Error: {e}"}
            
            # Step 4: Merge
            try:
                final_image = self._add_result_to_image(warped_img, mask, original_image_path)
            except Exception as e: 
                return {"status": "error", "message": f"Merging Error: {e}"}

            # Saving Result
            output_dir = os.path.dirname(original_image_path)
            temp_dir = os.path.join(output_dir, "temp_output")
            os.makedirs(temp_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.basename(original_image_path)
            name, ext = os.path.splitext(base_name)
            
            # Ensure proper extension handling
            final_filename = f"{name}_RENDERED_{timestamp}{ext if ext else '.jpg'}"
            final_path = os.path.join(temp_dir, final_filename)
            final_image.save(final_path,
                             quality=100,
                             subsampling=4)
            
            if self.debug: print(f"    [Success] Saved to: {final_path}")
            return {"status": "success", "rendered_image_path": final_path}
            
        except Exception as e:
            if self.debug: print(f"    [Error] Pipeline failed: {e}")
            return {"status": "error", "message": str(e)}
        


# =============================================================================
# TEST EXECUTION
# =============================================================================
if __name__ == "__main__":

    # Define test paths
    CROP = "src/document_image_renderer/test_document_image_renderer/test_data/preprocessed_test2_20251223_221622_document.jpg"
    ORIG = "src/document_image_renderer/test_document_image_renderer/test_data/preprocessed_test2_20251223_221622.jpg"
    JSON = "src/document_image_renderer/test_document_image_renderer/test_data/ocr/ocr_result.json" 

    LAYOUT = [
        {'label': 'top_left', 'x': 196.17242431640625, 'y': 94.59609985351562}, 
        {'label': 'top_right', 'x': 807.6567993164062, 'y': 94.59609985351562}, 
        {'label': 'bottom_right', 'x': 807.6567993164062, 'y': 888.10888671875}, 
        {'label': 'bottom_left', 'x': 196.17242431640625, 'y': 888.10888671875}] 
    try:
        renderer = ImageRenderer(debug=True)
        print("Starting Render Test...")
        print("\n=============================================================================")

        result = renderer.render_translated_image(
            cropped_image_path=CROP,
            original_image_path=ORIG,
            ocr_json_path=JSON,
            layout_coords=LAYOUT
        )
        print("Result:", result)
    except Exception as e:
        print("CRASH:", e)
        import traceback
        traceback.print_exc()