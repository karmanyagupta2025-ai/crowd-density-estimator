# CrowdSense — Crowd Density Estimator

A computer vision system that estimates the number of people in an image or video frame and generates a crowd density heatmap overlay. Built with YOLOv5 (primary) and YOLOv8 (experimental), FastAPI, OpenCV, and MongoDB.

***

## Demo

Upload a crowd image → receive a heatmap overlay + headcount estimate in the browser.  
Upload a crowd video → receive a processed MP4 with per-frame heatmap and crowd count burned in.

***

## Features

- Image upload → YOLO inference → JET colormap heatmap overlay + crowd count
- Video upload → per-5th-frame inference → H.264 re-encoded output with crowd count text overlay
- MongoDB logging of every prediction (filename, count, model, timestamp)
- Dual-model support: switch between YOLOv5 (stable) and YOLOv8 (experimental) via a single config flag
- Browser-compatible video output via FFmpeg H.264 re-encoding

***

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Inference | YOLOv5 via `torch.hub` / YOLOv8 via `ultralytics` |
| Image processing | OpenCV, Pillow |
| Database | MongoDB Atlas |
| Frontend | HTML + Bootstrap 5 + JavaScript |
| Video encoding | FFmpeg (libx264) |

***

## Dataset

Model trained on **ShanghaiTech Part A and Part B** crowd counting dataset.  
The YOLOv5 model (`best_yolov5.pt`) was fine-tuned on person detection annotations from ShanghaiTech.  
The YOLOv8 model (`best_yolov8.pt`) was trained by the team using the Ultralytics training pipeline.

***

## Model Architecture

### YOLOv5 (Primary — Stable)
- Base: YOLOv5s (small) custom-trained for single-class person detection
- Input resolution: 960×960
- Confidence threshold: 0.4, IoU threshold: 0.45
- Pre-processing: contrast enhancement (1.3×), Gaussian blur (r=1), sharpen
- Output: bounding boxes → centroid-based heatmap → JET colormap overlay

### YOLOv8 (Experimental)
- Base: YOLOv8 custom weights (`best_yolov8.pt`)
- Input: raw frame (no pre-processing)
- Confidence threshold: 0.1
- Status: integrated and functional, but currently undercounts vs YOLOv5 on test images — use for testing only

***

## Project Structure

```
crowd-density-estimator/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, /predict and /predict-video endpoints
│   │   ├── database.py          # MongoDB Atlas connection
│   │   └── services/
│   │       ├── predictor.py     # Image inference pipeline
│   │       └── video_predictor.py  # Video inference pipeline
│   ├── models/
│   │   ├── best_yolov5.pt       # YOLOv5 weights (primary, stable)
│   │   └── best_yolov8.pt       # YOLOv8 weights (experimental)
│   └── static/
│       └── cde_frontend.html    # Single-page browser UI
├── data/
│   └── samples/
│       └── test.jpg             # Sample input image
├── outputs/                     # Generated output videos (gitignored)
├── .env                         # MongoDB URI — NOT committed (see .env.example)
├── .gitignore
└── requirements.txt
```

***

## Setup Instructions

### Prerequisites

- Python 3.10+
- FFmpeg installed and available on PATH ([download here](https://ffmpeg.org/download.html))
- MongoDB Atlas account with a cluster URI

***

### 1. Clone the repository

```bash
git clone https://github.com/karmanyagupta2025-ai/crowd-density-estimator.git
cd crowd-density-estimator
```

### 2. Create and activate virtual environment

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Linux / Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
MONGO_URI=your_mongodb_atlas_connection_string
```

> ⚠️ Never commit `.env` to git. It is already in `.gitignore`.

### 5. Start the backend

```bash
uvicorn backend.app.main:app --reload --port 5000
```

Wait for:
```
Application startup complete.
```

### 6. Start the frontend (new terminal window)

```bash
cd backend/static
python -m http.server 3000
```

### 7. Open in browser

```
http://localhost:3000/cde_frontend.html
```

***

## Usage

### Image prediction

1. Click the upload zone and select a `.jpg`, `.png`, or `.webp` crowd image
2. Ensure the API endpoint shows `http://127.0.0.1:5000/predict`
3. Click **Generate Heatmap**
4. The heatmap overlay renders on the canvas and the crowd count is displayed

### Video prediction

1. Click the upload zone and select an `.mp4` video
2. The frontend auto-detects video and routes to `/predict-video`
3. Click **Generate Heatmap**
4. Wait for backend processing (30 seconds to several minutes depending on video length)
5. Once `HTTP 200 OK` appears in the backend terminal, the video player loads automatically

***

## Switching Models

In both `backend/app/services/predictor.py` and `backend/app/services/video_predictor.py`, change the flag at the top:

```python
MODEL_TYPE = "yolov5"   # stable, recommended
# MODEL_TYPE = "yolov8" # experimental — currently undercounts on test images
```

***

## API Endpoints

| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/` | — | Health check JSON |
| POST | `/predict` | Image file (multipart) | `{ count, heatmap_base64 }` |
| POST | `/predict-video` | Video file (multipart) | `{ message, output_video }` |
| GET | `/outputs/{filename}` | — | Processed video file |

***

## Sample Input / Output

Sample input image is available at `data/samples/test.jpg`.  
Run the app and upload this image to generate a sample output heatmap.

***

## Known Limitations

- YOLOv8 experimental weights currently undercount people compared to the YOLOv5 model on standard test images; YOLOv5 is recommended for demos
- Video processing runs on CPU only — processing time scales with video length and resolution
- `tlsAllowInvalidCertificates=True` is set in `database.py` as a temporary workaround for MongoDB Atlas TLS; this should be resolved before production deployment
- The `outputs/` folder is not auto-cleaned; old processed videos accumulate until manually deleted

***

## Team

Karmanya Gupta - Backend, inference pipeline, frontend integration  
Ishita Dokania - YOLOv8 model training, dataset preparation
Evani - Dataset preparation
Siddhant Jain and Sabhya - Frontend, Website Designing

***

## License

For educational and non-commercial use only.
