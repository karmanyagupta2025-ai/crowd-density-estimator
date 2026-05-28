import torch
from PIL import Image
from PIL import ImageEnhance, ImageFilter
import numpy as np
import cv2
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

        # Convert PIL image to OpenCV format
        img_cv = np.array(img)

        # Create empty heatmap
        heatmap = np.zeros((img_cv.shape[0], img_cv.shape[1]), dtype=np.float32)

        # Count persons
        predicted_count = 0

        # Generate density points
        for *box, conf, cls in results.xyxy[0]:

            if int(cls) == 0:
                predicted_count += 1

                x1, y1, x2, y2 = map(int, box)

                # Find center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw density point
                cv2.circle(heatmap, (center_x, center_y), 20, 1, -1)

        # Apply Gaussian blur
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

        # Normalize heatmap
        if np.max(heatmap)>0:
            heatmap = np.uint8(255 * heatmap / np.max(heatmap))
        else:
            heatmap=np.vint8(heatmap)
        # Apply color map
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay heatmap on image
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

        # Count detections
        predicted_count = 0

        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # person class
                predicted_count += 1

        # Convert numpy array to PIL image
        result_img = Image.fromarray(overlay)

        return result_img, predicted_count
    except Exception as e:
        print(f"Inference Error: {e}")
        raise
