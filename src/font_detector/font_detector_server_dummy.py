from fastmcp import FastMCP
from typing import Dict, Any, Tuple

mcp = FastMCP()

# Tool Interface & Implementing Dummy Data for testing purpose

@mcp.tool("Font-Detector")
def get_dummy_font_values(
        text_box_image: Any,
        text_box_size: Dict,
        text: str
) -> Tuple[str, float]:

    """
    Dummy implementation of font detection for the project setup

    :param text_box_image: Will receive an image of the text box - for now a dummy string will be used
    :param text_box_size: Dict with the size of the text box e.g. {"width": 100, "height": 100}
    :param text: The text which was detected by the Layout detector

    :return dummy_font_name: The name of the font detected
    :return dummy_font_size: The size of the font detected
    """

    print(f"[Font-Detector] Dummy-Request received")
    print(f"-> Text Box Image: type{text_box_image}")
    print(f"-> Text Box Size: {text_box_size}")
    print(f"-> Text: {text}")

    #Dummy-Return Values
    dummy_font_name = "Arial"
    dummy_font_size = 12.0

    print(f"-> Returning: {dummy_font_name}, {dummy_font_size}pt")

    return dummy_font_name, dummy_font_size


if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=8000)