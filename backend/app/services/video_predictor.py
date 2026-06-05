import cv2
import numpy as np
import torch
import warnings
from PIL import Image
warnings.filterwarnings('ignore')
#Load Model
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='backend/models/best.pt',
    force_reload=False
)

#Video Function
def process_video(video_path, output_path):

    #Open the Input Video
    cap = cv2.VideoCapture(video_path)

    #Get Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps<=0:
        fps=30


    #Define codec and output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0


    while True:

        #Read Frame
        ret, frame = cap.read()

        if not ret:
            break
        frame_count += 1

        #Processing the 5th frame only every time
        if frame_count % 5 !=0:
            continue

        # Run YOLO inference
        results = model(frame, size=640)

        # Create empty heatmap
        heatmap = np.zeros(
            (frame.shape[0], frame.shape[1]),
            dtype=np.float32
        )

        # Count people
        predicted_count = 0

        # Process detections
        for *box, conf, cls in results.xyxy[0]:

            # Class 0 = person
            if int(cls) == 0:
                predicted_count += 1

                x1, y1, x2, y2 = map(int, box)

                # Find center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw density point
                cv2.circle(
                    heatmap,
                    (center_x, center_y),
                    20,
                    1,
                    -1
                )

        # Apply Gaussian blur
        heatmap = cv2.GaussianBlur(
            heatmap,
            (51, 51),
            0
        )

        # Normalize safely
        if np.max(heatmap) > 0:
            heatmap = np.uint8(
                255 * heatmap / np.max(heatmap)
            )
        else:
            heatmap = np.uint8(heatmap)

        # Apply color map
        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # Overlay heatmap on frame
        overlay = cv2.addWeighted(
            frame,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        # Add crowd count text
        cv2.putText(
            overlay,
            f'Crowd Count: {predicted_count}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Write processed frame
        out.write(overlay)


        # Release resources

    cap.release()
    out.release()