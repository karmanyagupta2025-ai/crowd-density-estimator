import torch
from PIL import Image
from PIL import ImageEnhance, ImageFilter
import numpy as np
import io

# Load YOLOv5 model once
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='backend/models/best.pt',
    force_reload=True
)
model.conf=0.4
model.iou=0.45


def run_prediction(image_bytes: bytes):

    try:
        # Read uploaded image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        #Resize but keep aspect ratio
        img.thumbnail((960, 960))

        #Increase Contrast
        contrast=ImageEnhance.Contrast(img)
        img=contrast.enhance(1.3)

        #Gaussian Blur
        img=img.filter(ImageFilter.GaussianBlur(radius=1))

        #Sharpen Image
        img=img.filter(ImageFilter.SHARPEN)
        # Run YOLOv5 inference
        results = model(img, size=960)

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
    except Exception as e:
        print(f"Inference Error: {e}")
        raise
