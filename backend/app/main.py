from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.app.services.predictor import run_prediction
from backend.app.services.video_predictor import process_video
from backend.app.database import predictions_collection
from datetime import datetime

import base64
import io
import traceback
from PIL import Image

app = FastAPI()


# CORS — must be added BEFORE any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Static Files
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)

@app.get("/")
def health():
    return {"status": "CrowdSense backend running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received")

    try:
        # =====================================================
        # PLACEHOLDER — replace this block with your model
        result_img, count = run_prediction(contents)
        prediction_data={
            "filename": file.filename,
            "crowd_count": count,
            "model": "YOLOv5",
            "timestamp": datetime.utcnow()
        }
        predictions_collection.insert_one(prediction_data)
        # =====================================================

        buffer = io.BytesIO()
        result_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse({
            "count": count,
            "heatmap_base64": f"data:image/png;base64,{encoded}"
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    # Validate video file
    if not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Only video files are accepted"
        )

    try:

        # Save uploaded video
        input_path = f"outputs/{file.filename}"

        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Output processed video path
        output_path = f"outputs/processed_{file.filename}"

        # Run video processing
        process_video(input_path, output_path)

        return JSONResponse({
            "message": "Video processed successfully",
            "output_video": f"/outputs/processed_{file.filename}"
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )