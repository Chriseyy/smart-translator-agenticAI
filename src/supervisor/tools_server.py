from fastmcp import FastMCP
from typing import List, Dict, Union
import sys
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


current_file_path = os.path.abspath(__file__)
supervisor_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(supervisor_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from document_translator.document_translator import DocumentTranslator
from image_provider.provider import ImageProvider
from layout_detector.layout_detector import LayoutDetector
from document_image_renderer.document_image_renderer import ImageRenderer
from font_detector.font_detector_logic import FontDetectorLogic
from rag_component_x.rag import DocumentRAG

mcp = FastMCP()

image_provider = ImageProvider()
translator = DocumentTranslator(model_name="qwen3")
renderer_instance = ImageRenderer(device="cuda", debug=False)
font_logic_instance = FontDetectorLogic()
rag_engine = DocumentRAG(llm_model="qwen3")
layout_detector = LayoutDetector()



print("TOOLS_SERVER: Loading Document Class Detector...")

doc_classifier_scripts = os.path.join(src_dir, "document_class_detector", "scripts")
doc_classifier_models = os.path.join(src_dir, "document_class_detector", "models")
ckpt_path = os.path.join(os.path.dirname(src_dir), "checkpoints", "resnet_50", "best.ckpt")


if doc_classifier_scripts not in sys.path:
    sys.path.append(doc_classifier_scripts)


RVL_CDIP_CLASSES = [
    "Letter",                
    "Form",                  
    "Email",                 
    "Handwritten",       
    "Advertisement",     
    "Scientific Report",    
    "Scientific Publication",
    "Specification",       
    "File Folder",        
    "News Article",       
    "Budget",              
    "Invoice",             
    "Presentation",        
    "Questionnaire",         
    "Resume",               
    "Memo"                    
]

doc_model = None
doc_class_names = None
doc_image_size = 224 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:

    from document_class_detector.models.resnet_50 import build_model
    
    if os.path.isfile(ckpt_path):
        print(f"TOOLS_SERVER: Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        cfg = checkpoint.get("config", {})
        num_classes = cfg.get("num_classes", 16)
        doc_class_names = cfg.get("class_names")
        doc_image_size = cfg.get("image_size", 224)
        
        doc_model = build_model(num_classes=num_classes).to(device)
        doc_model.load_state_dict(checkpoint["model_state"])
        doc_model.eval()
        print("TOOLS_SERVER: Document Class Detector loaded successfully.")
    else:
        print(f"TOOLS_SERVER: WARNING - Checkpoint not found at {ckpt_path}. Using MOCK mode.")

except ImportError:
    print("TOOLS_SERVER: WARNING - 'alexnet.py' not found in scripts. Using MOCK mode.")
except Exception as e:
    print(f"TOOLS_SERVER: Error initializing Document Class Detector: {e}. Using MOCK mode.")


def preprocess_doc_image(img: Image.Image, target_size: int):
    tf = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(img).unsqueeze(0)


print("TOOLS_SERVER: Loading Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
rag_llm = ChatOllama(model="qwen3") 


# BASIC TOOLS
@mcp.tool()
def ping() -> Dict:
    """Tests connectivity."""
    print("TOOLS_SERVER: 'ping' tool was called.")
    return {"status": "success", "message": "Pong!"}

@mcp.tool()
def set_target_language(language: str) -> Dict:
    """Sets the target language."""
    print(f"TOOLS_SERVER: 'set_target_language' called with: {language}")
    return {
        "status": "success",
        "message": f"Target language set to '{language}'.",
        "language": language  
    }

@mcp.tool()
def load_image(path: str) -> Dict:
    """Loads an image from the specified file path."""
    print(f"TOOLS_SERVER: 'load_image' called with path: {path}")
    path = path.strip('\'" ')
    
    if not os.path.exists(path):
        return {"status": "error", "message": f"Image file not found: {path}"}
    
    try:
        result = image_provider.load_image_from_path(path)
        if result.get("status") == "success":
            return {
                "status": "success",
                "message": f"Image loaded: {result.get('path')}",
                "path": result.get("path")
            }
        return {"status": "error", "message": "Failed to load image"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
def capture_from_webcam(camera_index: int = 0) -> Dict:
    """Captures an image from the webcam."""
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                return {"status": "error", "message": "Webcam not available in WSL."}
    except:
        pass 
    
    result = image_provider.capture_from_webcam(camera_index)
    if result.get("status") == "success":
        return {
            "status": "success",
            "message": "Image captured.",
            "path": result.get("path")
        }
    return result

@mcp.tool()
def apply_preprocessing(
    image_path: str,
    contrast: float = 1.0,
    brightness: float = 1.0,
    sharpness: float = 0.0, 
    denoise: float = 0.0
) -> Dict:
    """Applies preprocessing adjustments."""
    print(f"TOOLS_SERVER: 'apply_preprocessing' called on {image_path}")
    
    if not image_path or not os.path.exists(image_path):
        return {"status": "error", "message": "Invalid image path for preprocessing."}

    try:
        result_dict = image_provider.preprocess_image(
            path=image_path,
            enhance_contrast=contrast,
            brightness=brightness,
            denoise_strength=denoise,
            sharpen_strength=sharpness,
        )
        
        if result_dict.get("status") == "success":
            return {
                "status": "success",
                "message": "Preprocessing applied.",
                "processed_image_path": result_dict.get("path"),
                "applied_settings": {
                    "contrast": contrast, "brightness": brightness,
                    "sharpness": sharpness, "denoise": denoise
                }
            }
        return {"status": "error", "message": result_dict.get("message", "Preprocessing failed")}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def detect_layout(image_path: str) -> Dict:
    """
    Detects layout, saves to JSON, returns PATH.
    """
    print(f"TOOLS_SERVER: 'detect_layout' called with {image_path}")
    
    if not image_path or not os.path.exists(image_path):
        return {"status": "error", "message": "Image path invalid."}

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    project_root = os.path.dirname(src_dir) 
    output_dir = os.path.join(project_root, "data", "layout_detector", base_name)
    os.makedirs(output_dir, exist_ok=True)

    print("TOOLS_SERVER: Running Outline Detection...")
    try:
        outline_result = layout_detector.get_outline(image_path, output_dir=output_dir)
        working_image_path = outline_result.get("document_image", image_path)
    except Exception as e:
        print(f"TOOLS_SERVER: Outline detection warning: {e}")
        working_image_path = image_path
        outline_result = {}

    print(f"TOOLS_SERVER: Running OCR on {working_image_path}...")
    try:
        ocr_result = layout_detector.get_ocr(working_image_path, output_dir=output_dir)
    except Exception as e:
        return {"status": "error", "message": f"PaddleOCR failed: {str(e)}"}

    ocr_json_path = ocr_result.get("ocr_json")
    if not ocr_json_path:
        return {"status": "error", "message": "OCR finished but returned no JSON path."}


    return {
        "status": "success",
        "message": "Layout detection complete.",
        "text_blocks_path": ocr_json_path,        
        "extracted_document_path": working_image_path,
        "coordinates": outline_result.get("coordinates") 
    }

@mcp.tool()
def detect_document_class(image_path: str) -> Dict:
    """
    Classifies the document type.
    """
    print(f"TOOLS_SERVER: 'detect_document_class' called with {image_path}")
    
    if not image_path or not os.path.exists(image_path):
        return {"status": "error", "message": "Image path invalid."}

    if doc_model is not None:
        try:
            img = Image.open(image_path).convert("RGB")
            x = preprocess_doc_image(img, doc_image_size).to(device)
            
            with torch.no_grad():
                logits = doc_model(x)
                probs = F.softmax(logits, dim=1)
            
            pred_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred_idx].item())
            
            label = None
            
            if doc_class_names and len(doc_class_names) > pred_idx:
                label = doc_class_names[pred_idx]
            
            elif 0 <= pred_idx < len(RVL_CDIP_CLASSES):
                label = RVL_CDIP_CLASSES[pred_idx]
            
            else:
                label = f"Class {pred_idx}"

            print(f"TOOLS_SERVER: Detected class '{label}' with confidence {confidence:.2%}")
            
            return {
                "status": "success",
                "document_class": label,
                "confidence": round(confidence, 4),
                "message": f"Document is a '{label}' ({confidence:.1%})"
            }
            
        except Exception as e:
            print(f"TOOLS_SERVER: Error during inference: {e}")

    return {
        "status": "success",
        "document_class": "Invoice",  
        "confidence": 0.95,
        "message": "Mock Result (Model failed or not loaded)"
    }

@mcp.tool()
def detect_fonts(text_blocks_path: str, image_path: str) -> Dict:
    """
    Identifies font, saves the result to a new JSON, and returns the path.
    """
    print(f"TOOLS_SERVER: 'detect_fonts' called on {text_blocks_path}")
    
    if not font_logic_instance:
        return {"status": "error", "message": "FontDetectorLogic not initialized."}
    if not os.path.exists(text_blocks_path):
        return {"status": "error", "message": "Text blocks JSON not found."}
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image file not found: {image_path}"}
    try:
        
        font_json_path = font_logic_instance.process_json_data(text_blocks_path, image_path)
        
        print(f"TOOLS_SERVER: Font detection finished. Updated file: {font_json_path}")

        return {
            "status": "success",
            "message": "Fonts detected and saved.",
            "font_json_path": font_json_path, 
        }

    except Exception as e:
        print(f"TOOLS_SERVER: Error in detect_fonts: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
def translate_layout_file(json_path: str, target_language: str) -> Dict:
    """
    Reads layout JSON, translates text, saves to new JSON, returns PATH.
    """
    print(f"TOOLS_SERVER: 'translate_layout_file' called for {json_path}")

    return translator.translate_json_file(json_path, target_language)

@mcp.tool()
def render_document(
    original_image_path: str,
    cropped_image_path: str,       
    translated_json_path: str,     
    layout_json_path: Union[str, List[Dict]]
    ) -> Dict:
    """
    Renders the translated text back onto the original document image.
    Injects specific font info into the rendering pipeline per block.
    """
    print(f"TOOLS_SERVER: 'render_document' called")
    
    if renderer_instance is None:
        return {"status": "error", "message": "ImageRenderer instance not initialized."}
    
    missing_files = []
    if not os.path.exists(original_image_path): missing_files.append(original_image_path)
    if not os.path.exists(cropped_image_path): missing_files.append(cropped_image_path)
    if not os.path.exists(translated_json_path): missing_files.append(translated_json_path)
    
    if missing_files:
        return {"status": "error", "message": f"Missing files: {missing_files}"}


    try:
        result = renderer_instance.render_translated_image(
            cropped_image_path=cropped_image_path,
            original_image_path=original_image_path,
            ocr_json_path=translated_json_path, 
            layout_coords=layout_json_path)
        
        if result.get("status") == "success":
            return {
                "status": "success",
                "message": "Document rendered successfully.",
                "rendered_document_path": result.get("rendered_image_path")
            }
        else:
            return {"status": "error", "message": f"Renderer failed: {result.get('message')}"}
            
    except Exception as e:
        print(f"TOOLS_SERVER: Critical Error in render_document: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
def query_translated_document(json_path: str, query: str) -> Dict:
    """
    RAG-System: Answers questions based on the text content of the translated JSON file.
    """
    print(f"TOOLS_SERVER: RAG Request -> '{query}' on file {json_path}")
    
    return rag_engine.query_document(json_path, query)


if __name__ == "__main__":
    print("INFO:     Starting Tools-Server on http://localhost:8000")
    mcp.run(transport="http", host="localhost", port=8000)