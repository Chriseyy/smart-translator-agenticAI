import csv
import random
import string
from PIL import ImageFont, Image, ImageDraw
import os

def generate_training_data(
    num_samples: int,
    min_width: int,
    max_width: int,
    min_font_size: int,
    max_font_size: int,
    font_path: str,
    output_file: str,
):
    """
    Generate a synthetic dataset for training a model to predict the maximum font size
    that allows a given random text string to fit into a bounding box of given width.

    Each row in the output CSV contains:
        - box_width: the width of the bounding box (random between min_width and max_width)
        - box_height: currently unused (set to 0, placeholder for possible future extension)
        - text_length: the number of characters in the generated text
        - char_<char>_count: one column per character in the allowed character set,
                             counting occurrences of that character in the text
        - font_size: the largest font size (≥ min_font_size, ≤ max_font_size)
                     such that the text fits within the box_width
        - text: the actual random text string

    Args:
        num_samples (int): Number of rows/samples to generate.
        min_width (int): Minimum width of the bounding box.
        max_width (int): Maximum width of the bounding box.
        min_font_size (int): Smallest font size considered when fitting text.
        max_font_size (int): Largest font size considered when fitting text.
        font_path (str): Path to a TrueType font file (e.g. "C:/Windows/Fonts/arial.ttf").
        output_file (str): Path to the CSV file where results will be stored.

    Raises:
        FileNotFoundError: If the font file cannot be found or loaded.

    Notes:
        - Only text width is considered when fitting the font size. Height is currently unused.
        - The character set includes ASCII letters and German umlauts (äöüÄÖÜ).
    """
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")

    # Character set: ASCII letters + umlauts
    standard_letters = string.ascii_letters
    umlauts = "äöüÄÖÜ"
    chars = standard_letters + umlauts

    # Build headers dynamically from the actual character set
    char_headers = [f"char_{ch}_count" for ch in chars]
    headers = ["box_width", "box_height", "text_length"] + char_headers + ["font_size", "text"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(num_samples):
            box_width = random.randint(min_width, max_width)
            box_height = 0
            text_length = random.randint(5, 50)

            text = "".join(random.choice(chars) for _ in range(text_length))

            # Initialize character counts
            char_counts = {ch: 0 for ch in chars}
            for ch in text:
                if ch in char_counts:
                    char_counts[ch] += 1

            final_fs, box_width, box_height = find_max_font_size(text, box_width, box_height, font_path, min_font_size, max_font_size)

            """
            font = ImageFont.truetype(font_path, final_fs)
            img = Image.new("RGB", (500 * 2, 500 * 2), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), text, font=font, fill="black")
            draw.rectangle([0, 0, box_width, box_height],outline="blue", width=1)
            img.show()
            """

            # Build the feature vector
            feature_vector = [box_width, box_height, text_length]
            feature_vector.extend([char_counts[ch] for ch in chars])
            feature_vector.append(final_fs)
            feature_vector.append(text)

            writer.writerow(feature_vector)

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} samples...")

    print("Training generators generation complete!")

def find_max_font_size(text, box_width, box_height, font_path,
                       min_fs=5, max_fs=200, precision=0.1):

    def get_box_fast(font, text):
        bbox = font.getbbox(text)
        return bbox[2], bbox[3]

    def get_box_precise(font, text):
        img = Image.new("L", (box_width, box_height), 0)  # big enough canvas
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font, fill=255)
        bbox = img.getbbox()  # actual non-empty pixels
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def get_box_super_precise(font, text):
        # make a large enough canvas so text won’t clip
        img = Image.new("L", (box_width * 2, box_width * 2), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font, fill=255)

        # find the exact bounding box of nonzero pixels
        bbox = img.getbbox()
        if bbox is None:
            return 0, 0  # empty text
        x0, y0, x1, y1 = bbox
        return x1 - x0, y1 - y0

    def get_box_via_mask(font, text):
        return font.getmask(text).size

    def measure(font_size):
        font = ImageFont.truetype(font_path, int(round(font_size)))
        return get_box_fast(font, text)

    low, high = min_fs, max_fs
    best = min_fs
    text_w = 0
    text_h = 0

    while high - low > precision:
        mid = (low + high) / 2
        text_w, text_h = measure(mid)

        if text_w <= box_width: # and text_h <= box_height:
            best = mid
            low = mid  # try larger
        else:
            high = mid  # too big → shrink

    return round(best, 2), text_w, text_h

#font_path = r"C:\Windows\Fonts\arial.ttf" # path is necessary in Windows.

#generate_training_data(num_samples=10000, min_width=100, max_width=800, min_font_size=4,
#                       max_font_size=32, font_path=font_path, output_file='training_data.csv')