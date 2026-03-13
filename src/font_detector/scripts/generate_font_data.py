import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import string
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

# Font names
FONT_NAMES = ["Arial", "Times New Roman", "Verdana", "Courier New", "Comic Sans MS"]

# Font file names
FONT_FILES = {
    "Arial": "arial.ttf",
    "Times New Roman": "times.ttf",
    "Verdana": "verdana.ttf",
    "Courier New": "cour.ttf",
    "Comic Sans MS": "comic.ttf"
}

# Path to font directory
FONT_DIR = Path(r"C:\Windows\Fonts")

# All characters to be used for training
CHARACTERS = string.ascii_letters + string.digits + " .,!?äöüÄÖÜß"

# Number of samples per font and define range of font sizes and text lengths v
NUM_SAMPLES_PER_FONT = 5000
MIN_FONT_SIZE = 8.0
MAX_FONT_SIZE = 72.0
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 100


def get_features_for_text(text: str, font_path: Path, font_size: float) -> dict[str, int | float] | None:
    """
    Returns a dictionary of features for a given text and font.

    Args:
        text (str): The text to extract features from.
        font_path (Path): The path to the font file.
        font_size (float): The font size in points.

    returns:
        dict[str, int | float]: A dictionary of features for the text and font.
        None: If the font could not be loaded.
    """
    try:
        font = ImageFont.truetype(str(font_path), size=int(round(font_size)))
    except IOError:
        print(f"Warning: Could not load font '{font_path}'.")
        return None

    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    bbox = draw.textbbox((0, 0), text, font=font)
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]

    features = {"text_length": len(text), "box_width": box_width, "box_height": box_height}

    char_counts = Counter(text)
    for char in CHARACTERS:
        features[f"dist_{char}"] = char_counts.get(char, 0)

    return features


def generate_dataset():
    """
    Generates dataset based on defined font names, character constraints, and other configuration
    parameters. Random text, font size, and their respective features with metadata are created
    and saved into Training, Validation, and Test datasets.
    """
    all_data = []

    for font_name in FONT_NAMES:
        print(f"Generating data for: {font_name}...")
        font_file = FONT_FILES.get(font_name)
        if not font_file:
            print(f"No .ttf file defined for {font_name}. Skipping.")
            continue

        font_path = FONT_DIR / font_file
        if not font_path.exists():
            print(f"Font file not found: {font_path}. Skipping.")
            continue

        for i in range(NUM_SAMPLES_PER_FONT):
            # Generate random values
            text_len = random.randint(MIN_TEXT_LENGTH, MAX_TEXT_LENGTH)
            # Create a random string from the allowed characters
            text = "".join(random.choices(CHARACTERS, k=text_len))

            # Generate a random font size
            font_size = random.uniform(MIN_FONT_SIZE, MAX_FONT_SIZE)

            # Extract features
            features = get_features_for_text(text, font_path, font_size)
            if not features:
                continue

            # Add labels and metadata
            features["font_name"] = font_name
            features["font_size_label"] = font_size

            all_data.append(features)

    # --- 5. Save and Split ---
    if not all_data:
        print("No data generated!")
        return

    df = pd.DataFrame(all_data)

    # Create a folder for data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Split into Training, Validation, and Test sets
    # First, split off the Test set (15% of total data)
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    # Then split the remaining 85% into Training and Validation (10% of the remaining)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

    # Save dataframes to CSV files
    train_df.to_csv(data_dir / "font_size_train.csv", index=False)
    val_df.to_csv(data_dir / "font_size_val.csv", index=False)
    test_df.to_csv(data_dir / "font_size_test.csv", index=False)

    print(f"Datasets successfully generated and saved to '{data_dir}'.")
    print(f"Training:   {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")
    print(f"Test:       {len(test_df)} rows")


if __name__ == "__main__":
    generate_dataset()