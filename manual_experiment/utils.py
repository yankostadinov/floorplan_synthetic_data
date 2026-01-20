import base64
import os
from pathlib import Path


def list_filenames(path):
    return [
        Path(f).stem for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    ]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def plan_to_base64(plan_name):
    image_path = f"plans/{plan_name}.png"
    encoded_image = encode_image(image_path)
    return f"data:image/png;base64,{encoded_image}"


def save_text_to_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as myfile:
        myfile.write(text)
