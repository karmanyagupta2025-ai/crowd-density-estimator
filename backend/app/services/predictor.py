import torch
from PIL import Image
import numpy as np
import io

# Load YOLOv5 model once
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='backend/models/best.pt',
    force_reload=True
)


def run_prediction(image_bytes: bytes):

    # Read uploaded image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run YOLOv5 inference
    results = model(img)

    # Render bounding boxes on image
    results.render()

    # Get rendered image
    rendered_img = results.ims[0]

    # Count detections
    predicted_count = 0

    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # person class
            predicted_count += 1

    # Convert numpy array to PIL image
    result_img = Image.fromarray(rendered_img)

    return result_img, predicted_count