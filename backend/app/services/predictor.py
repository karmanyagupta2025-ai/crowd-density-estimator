import torch
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io

# Switch between models: "yolov5" or "yolov8"
MODEL_TYPE = "yolov5"

if MODEL_TYPE == "yolov8":
    from ultralytics import YOLO
    model = YOLO("backend/models/best_yolov8.pt")
else:
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path='backend/models/best_yolov5.pt',
        force_reload=False
    )
    model.conf = 0.4
    model.iou = 0.45


def run_prediction(image_bytes: bytes):
    try:
        # Read uploaded image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize but keep aspect ratio
        img.thumbnail((960, 960))

        if MODEL_TYPE == "yolov5":
            # Pre-processing for YOLOv5
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(1.3)
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            img = img.filter(ImageFilter.SHARPEN)

        # Convert PIL image to OpenCV format
        img_cv = np.array(img)

        # Create empty heatmap
        heatmap = np.zeros((img_cv.shape[0], img_cv.shape[1]), dtype=np.float32)
        predicted_count = 0

        if MODEL_TYPE == "yolov8":
            results = model(img_cv, conf=0.1)[0]
            boxes_list = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            for box in boxes_list:
                predicted_count += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.circle(heatmap, ((x1 + x2) // 2, (y1 + y2) // 2), 20, 1, -1)
        else:
            results = model(img, size=960)
            for *box, conf, cls in results.xyxy[0]:
                if int(cls) == 0:
                    predicted_count += 1
                    x1, y1, x2, y2 = map(int, box)
                    cv2.circle(heatmap, ((x1 + x2) // 2, (y1 + y2) // 2), 20, 1, -1)

        # Apply Gaussian blur to heatmap
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = np.uint8(255 * heatmap / np.max(heatmap))
        else:
            heatmap = np.uint8(heatmap)

        # Apply color map
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay heatmap on image
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

        # Convert to PIL image
        result_img = Image.fromarray(overlay)

        return result_img, predicted_count

    except Exception as e:
        print(f"Inference Error: {e}")
        raise