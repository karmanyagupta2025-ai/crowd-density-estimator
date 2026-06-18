import cv2
import numpy as np
import torch
import subprocess
import os
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

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


def process_video(video_path, output_path):

    temp_path = output_path.replace(".mp4", "_temp.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_count = 0
    last_overlay = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            predicted_count = 0

            if MODEL_TYPE == "yolov8":
                results = model(frame, conf=0.1)[0]
                boxes_list = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
                for box in boxes_list:
                    predicted_count += 1
                    x1, y1, x2, y2 = map(int, box)
                    cv2.circle(heatmap, ((x1 + x2) // 2, (y1 + y2) // 2), 20, 1, -1)
            else:
                results = model(frame, size=640)
                for *box, conf, cls in results.xyxy[0]:
                    if int(cls) == 0:
                        predicted_count += 1
                        x1, y1, x2, y2 = map(int, box)
                        cv2.circle(heatmap, ((x1 + x2) // 2, (y1 + y2) // 2), 20, 1, -1)

            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

            if np.max(heatmap) > 0:
                heatmap = np.uint8(255 * heatmap / np.max(heatmap))
            else:
                heatmap = np.uint8(heatmap)

            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
            cv2.putText(
                overlay,
                f'Crowd Count: {predicted_count}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            last_overlay = overlay

        else:
            overlay = last_overlay if last_overlay is not None else frame

        out.write(overlay)

    cap.release()
    out.release()

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ], capture_output=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg re-encode failed:\n{result.stderr.decode()}"
        )