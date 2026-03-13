import sys
import pickle
import numpy as np
import string
import base64
import io
import json
import torch
import shutil
import os
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load tiny_diff
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

if str(current_file.parent) not in sys.path:
    sys.path.append(str(current_file.parent))

from tiny_diff.scalar.io import load_model_weights
from tiny_diff.scalar.node import Node

try:
    from src.font_detector.model import FontSizeMLP
except ImportError:
    try:
        from model import FontSizeMLP
    except ImportError:
        pass


class FontDetectorLogic:
    def __init__(self):
        print("[Logic] Initialize Font Detector System...")
        self.base_dir = current_file.parent
        self.models_dir = self.base_dir.parent.parent / "models"

        # Fallback
        if not self.models_dir.exists():
            self.models_dir = self.base_dir / "models"

        #JSON Path
        self.ocr_dir = project_root / "example_files" / "layout_detector" / "get_ocr"

        self.fonts_dir = project_root / "assets" / "fonts"
        self.default_font_path = str(self.fonts_dir / "arial.ttf")

        self.font_path_map = {
            # Cluster Arial
            "Arial": str(self.fonts_dir / "arial.ttf"),

            # Cluster Times New Roman
            "Times New Roman": str(self.fonts_dir / "times.ttf"),

            # Cluster Verdana
            "Verdana": str(self.fonts_dir / "verdana.ttf"),

            # Cluster Courier New
            "Courier New": str(self.fonts_dir / "cour.ttf"),

            # Cluster Comic Sans
            "Comic Sans MS": str(self.fonts_dir / "comic.ttf"),
        }

        # Font Name Detection Modell
        self.phase1_ready = False
        try:
            p1_path = self.models_dir / "font_name_detector"
            print(f"Load Font Name Detection from: {p1_path}")
            if (p1_path / "config.json").exists():
                self.processor = AutoImageProcessor.from_pretrained(str(p1_path))
                self.cnn_model = AutoModelForImageClassification.from_pretrained(str(p1_path))
                self.phase1_ready = True
                print("Font Name Detection loaded.")
            else:
                print("Modell-Pfad existiert nicht.")
        except Exception as e:
            print(f"Font Name Detection not loaded (Fallback: Arial): {e}")

        # Font Size estimation Models
        print("   Load Phase 2 (Size MLPs)...")
        self.fonts = ["Arial", "Times New Roman", "Verdana", "Courier New", "Comic Sans MS"]
        self.size_models = {}
        self.scalers = {}

        dummy_dim = self._extract_features("test", 100, 20).shape[1]
        self.size_models_dir = self.models_dir / "font_size"

        for font in self.fonts:
            try:
                scaler_path = self.size_models_dir / f"{font}_scaler.pkl"
                model_path = self.size_models_dir / f"{font}_model.json"

                if scaler_path.exists() and model_path.exists():
                    with open(scaler_path, "rb") as f:
                        self.scalers[font] = pickle.load(f)

                    model = FontSizeMLP(dummy_dim)
                    load_model_weights(model, str(model_path))
                    self.size_models[font] = model
            except Exception as e:
                # print(f"Warning: Failed to load size model for {font}: {e}")
                pass

        print(f"Font Size: ({len(self.size_models)} Models loaded).")

        # --- CLUSTER MAPPING ---
        self.font_cluster_map = {
            # CLUSTER: TIMES NEW ROMAN
            "Times New Roman": "Times New Roman",
            "Times New Roman Bold": "Times New Roman",
            "Times New Roman Italic": "Times New Roman",
            "Times New Roman Bold Italic": "Times New Roman",
            "Georgia": "Times New Roman",
            "Lora-Regular": "Times New Roman",
            "Merriweather-Regular": "Times New Roman",
            "PlayfairDisplay-Regular": "Times New Roman",
            "RobotoSlab-Regular": "Times New Roman",

            # CLUSTER: ARIAL
            "Arial": "Arial",
            "Arial Bold": "Arial",
            "Arial Black": "Arial",
            "Arial Bold Italic": "Arial",
            "Helvetica": "Arial",
            "Avenir": "Arial",
            "Inter-Regular": "Arial",
            "Roboto-Regular": "Arial",
            "Lato-Regular": "Arial",
            "Poppins-Regular": "Arial",
            "IBMPlexSans-Regular": "Arial",

            # CLUSTER: VERDANA
            "Verdana": "Verdana",
            "Verdana Bold": "Verdana",
            "Verdana Italic": "Verdana",
            "Verdana Bold Italic": "Verdana",
            "Tahoma": "Verdana",
            "Tahoma Bold": "Verdana",
            "Trebuchet MS": "Verdana",
            "Trebuchet MS Bold": "Verdana",
            "Trebuchet MS Italic": "Verdana",
            "Trebuchet MS Bold Italic": "Verdana",
            "OpenSans-Bold": "Verdana",
            "OpenSans-Italic": "Verdana",
            "OpenSans-Light": "Verdana",
            "Rubik-Regular": "Verdana",
            "TitilliumWeb-Regular": "Verdana",

            # CLUSTER: COURIER NEW
            "Courier": "Courier New",
            "Courier New": "Courier New",
            "RobotoMono-Regular": "Courier New",
            "SpaceMono-Regular": "Courier New",

            # CLUSTER: COMIC SANS MS
            "Comic Sans MS": "Comic Sans MS",
            "Agbalumo-Regular": "Comic Sans MS",
            "AlfaSlabOne-Regular": "Comic Sans MS",
            "ArchitectsDaughter-Regular": "Comic Sans MS",
            "Bangers-Regular": "Comic Sans MS",
            "BlackOpsOne-Regular": "Comic Sans MS",
            "KaushanScript-Regular": "Comic Sans MS",
            "Lobster-Regular": "Comic Sans MS",
            "Niconne-Regular": "Comic Sans MS",
            "Pacifico-Regular": "Comic Sans MS",
            "PixelifySans-Regular": "Comic Sans MS",
            "Rakkas-Regular": "Comic Sans MS"
        }

    def _extract_features(self, text, width, height):
        features = [float(width), float(height), float(len(text))]

        for char in string.ascii_lowercase:
            features.append(float(text.count(char)))

        for char in string.ascii_uppercase:
            features.append(float(text.count(char)))

        for char in string.digits:
            features.append(float(text.count(char)))

        specials = ".,!?:;\"'()-_"  # 12 Zeichen

        for char in specials:
            features.append(float(text.count(char)))

        current_len = len(features)
        target_len = 77

        if current_len < target_len:
            features.extend([0.0] * (target_len - current_len))
        elif current_len > target_len:
            features = features[:target_len]

        return np.array([features])

    def predict(self, image_input, width: int, height: int, text: str):
        """
        Run pipeline.
        image_input: Can be a Base64 string OR a PIL Image object.
        """
        # 1. Font name recognition
        detected_name = "Arial"  # Default Fallback

        if self.phase1_ready:
            try:
                image = None
                # Check input type
                if isinstance(image_input, str):
                    if len(image_input) > 10:
                        if "," in image_input:
                            image_input = image_input.split(",")[1]
                        image_data = base64.b64decode(image_input)
                        image = Image.open(io.BytesIO(image_data)).convert("RGB")
                elif isinstance(image_input, Image.Image):
                    image = image_input.convert("RGB")

                if image:
                    inputs = self.processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.cnn_model(**inputs)

                    idx = outputs.logits.argmax(-1).item()

                    try:
                        detected_name = self.cnn_model.config.id2label[idx]
                    except KeyError:
                        detected_name = self.cnn_model.config.id2label[str(idx)]

            except Exception as e:
                print(f"Error Font Name: {e}")

        # Cluster Mapping & Model Selection
        target_model_name = self.font_cluster_map.get(detected_name, "Arial")  # Fallback to Arial

        if target_model_name not in self.size_models:
            target_model_name = "Arial"

        # 3. Size Estimation
        final_size = 12.0  # Default

        if target_model_name in self.size_models:
            model = self.size_models[target_model_name]
            scaler = self.scalers[target_model_name]

            # Features -> Normalize -> Node -> Model -> Rescale
            raw = self._extract_features(text, width, height)
            norm = (raw[0] - scaler["mean"]) / (scaler["std"] + 1e-8)
            nodes = [Node(val) for val in norm]
            out_node = model(nodes)[0]

            final_size = out_node.value * scaler.get("y_scale", 100.0)

        return detected_name, float(final_size)

    def process_json_data(self, json_path: str, image_path: str) -> str:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        json_data["input_path"] = image_path

        # BackUp Input data
        output_dir = os.path.dirname(json_path)
        debug_dir = os.path.join(output_dir, ".backups")
        os.makedirs(debug_dir, exist_ok=True)
        backup_filename = f"input_font_detector_{os.path.basename(json_path)}"
        backup_path = os.path.join(debug_dir, backup_filename)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"[Backup] Created backup at: {backup_path}")
        
        # load picture
        input_path = json_data.get("input_path")

        if not input_path or not os.path.exists(input_path):
            print(f"[Error] File path not found: {input_path}")
            full_image = None
        else:
            try:
                full_image = Image.open(input_path).convert("RGB")
                print(f"[Process] Load pciture: {input_path}")
            except Exception as e:
                print(f"[Error] Picture can't be loaded: {e}")
                full_image = None

        # font detection
        rec_texts = json_data.get("rec_texts", [])
        rec_boxes = json_data.get("rec_boxes", [])

        detected_fonts = []
        detected_sizes = []

        print(f"[Process] Analyse {len(rec_texts)} text boxes...")

        for i, (text, box) in enumerate(zip(rec_texts, rec_boxes)):
            # Box Format: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            # Crop image logic
            crop_img = None
            if full_image:
                try:
                    crop_box = (
                        max(0, xmin),
                        max(0, ymin),
                        min(full_image.width, xmax),
                        min(full_image.height, ymax)
                    )
                    crop_img = full_image.crop(crop_box)
                except Exception as e:
                    print(f"Error cropping box {i}: {e}")

            font_name, font_size = self.predict(crop_img, width, height, text)

            font_path = self.font_path_map.get(font_name, self.default_font_path)

            detected_fonts.append(font_path)
            detected_sizes.append(round(font_size, 2))

        json_data["rec_font_names"] = detected_fonts
        json_data["rec_font_sizes"] = detected_sizes

        with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"[Save] Successfully updated: {json_path}")
        
        print("[Process] Done.")
        return json_path


if __name__ == "__main__":
    detector = FontDetectorLogic()

    base_dir = detector.ocr_dir

    # path to ocr json
    input_json_path = base_dir / "ocr_result.json"

    # path to image
    input_image_path = base_dir / "input_image.jpg"

    # checks
    if not input_json_path.exists():
        print(f"[Error] Input JSON file not found:\n{input_json_path}")
        sys.exit(1)
    
    if not input_image_path.exists():
        print(f"[Warning] Test Image not found:\n{input_image_path}")
        print("The detector might fail or print errors regarding the missing image.")

    try:
        print("--- START TEST ---")
        print(f"[Main] JSON:  {input_json_path}")
        print(f"[Main] Image: {input_image_path}")

        updated_path = detector.process_json_data(str(input_json_path), str(input_image_path))

        print("-" * 30)
        print(f"[Main] Successfully completed.")
        print(f"[Main] The file was updated in-place: {updated_path}")
        print(f"[Main] A backup should be in the .backups folder.")

    except json.JSONDecodeError:
        print(f"[Error] The file {input_json_path} is not a valid JSON file.")
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")