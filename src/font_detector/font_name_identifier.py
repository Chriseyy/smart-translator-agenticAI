# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Local file path to the model to identify the font name
local_model_path = r"src/font_detector/models/font_name_detector"

# Load model and processor
processor = AutoImageProcessor.from_pretrained(local_model_path)
model = AutoModelForImageClassification.from_pretrained(local_model_path)
print("Model and processor loaded")

# Load test picture
try:
    image = Image.open(r".\src\font_detector\font_image_samples\Arial_20Bold_39.png")
except FileNotFoundError:
    print("Image not found")
    exit()

# Start prediction
# Prepares the image for the model
inputs = processor(images=image, return_tensors="pt")

# Predicts the font name
outputs = model(**inputs)

# Extracts the predicted class index and get the best prediction
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

# Get the font name from the predicted class index to get the human-readable font name
font_name = model.config.id2label[predicted_class_idx]

print(f"Font name: {font_name}")

