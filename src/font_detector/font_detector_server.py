from fastmcp import FastMCP
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from font_detector_logic import FontDetectorLogic

mcp = FastMCP("Font-Detector-Pro")

print("--- Starte Server ---")
detector = FontDetectorLogic()
print("--- Server läuft ---")


@mcp.tool("detect_font_full")
def detect_font_full(
        text_box_image: str,
        text_box_size: Dict[str, int],
        text: str
) -> Dict[str, Any]:
    """
    Detects the font family and font size of a text box.

    Args:
        text_box_image: Base64 encoded string of the text box image crop.
        text_box_size: Dictionary containing 'width' and 'height' in pixels.
        text: The string content inside the text box.

    Returns:
        JSON object with 'font_family' (str) and 'font_size_pt' (float).
    """

    w = text_box_size.get("width", 100)
    h = text_box_size.get("height", 20)

    print(f"[Request] Text: '{text}' | Box: {w}x{h}")

    name, size = detector.predict(
        b64_image=text_box_image,
        width=w,
        height=h,
        text=text
    )

    print(f"[Result]  Font: {name} | Size: {size:.2f}pt")

    return {
        "font_family": name,
        "font_size_pt": round(size, 2),
        "confidence": "high" if name != "Arial" else "fallback"
    }


if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=8000)