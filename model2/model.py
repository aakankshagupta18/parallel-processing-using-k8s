from transformers import pipeline
from PIL import Image
import requests
import sys
import os
from io import BytesIO

def is_url(path):
    return path.startswith("http://") or path.startswith("https://")


def depth_estimation(image_source):
    
    if is_url(image_source):
        response = requests.get(image_source)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif os.path.exists(image_source):
        image = Image.open(image_source).convert("RGB")
    else:
        print("Error: Invalid image path or URL")
        sys.exit(1)

    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")

    # inference
    depth = pipe(image)
    print(depth['predicted_depth'])
    return depth['predicted_depth']

if len(sys.argv) < 2:
        print("Usage: python model.py <image_path_or_url>")
        sys.exit(1)

image_source = sys.argv[1]
depth_estimation(image_source)
