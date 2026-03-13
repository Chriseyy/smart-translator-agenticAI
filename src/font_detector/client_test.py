import asyncio
import json
import base64
from pathlib import Path
from fastmcp import Client

MCP_SERVER_URL = "http://localhost:8000/mcp"

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "font_image_samples"


def load_image_as_base64(filename: str) -> str:
    """
    Load an image from the image directory and encode it as Base64.
    """
    file_path = IMAGE_DIR / filename

    if not file_path.exists():
        file_path = BASE_DIR / filename

    if not file_path.exists():
        print(f"File '{filename}' not found in {IMAGE_DIR}")
        print("Sending Dummy Data")
        return "DummyBase64String=="

    print(f"Load file: {file_path.name}")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


async def test_font_detector_endpoint():
    print("Test Client")

    client = Client(MCP_SERVER_URL)

    image_filename = "Arial_20Bold_39.png"
    b64_image = load_image_as_base64(image_filename)

    inputs = {
        "text_box_image": b64_image,
        "text_box_size": {
            "width": 150,
            "height": 30
        },
        "text": "Hello World"
    }

    print(f"Verbinde zu Server: {MCP_SERVER_URL}")

    try:
        async with client:
            print(f"\n[Client] Call MCP Tool 'detect_font_full'")

            result = await client.call_tool("detect_font_full", inputs)

            data = None

            if isinstance(result, dict):
                data = result
            elif hasattr(result, 'content'):
                data = result.content


            if data:
                if hasattr(data, "type") and data.type == "text":
                    data = json.loads(data.text)

                if isinstance(data, dict):
                    font_name = data.get("font_family", "Unbekannt")
                    font_size = data.get("font_size_pt", 0.0)
                    confidence = data.get("confidence", "N/A")

                    print(f"Success")
                    print(f"Detected font name: {font_name}")
                    print(f"Estimated font size: {font_size} pt")
                    print(f"Status: {confidence}")
                else:
                    print(f"Raw data: {data}")
            else:
                print(f"Raw result: {result}")

    except (OSError, ConnectionError):
        print("\nError: Server not reachable.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(test_font_detector_endpoint())