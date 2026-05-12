from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2

from app.schemas.prediction import PredictionResponse
from app.services.predictor import run_prediction

router = APIRouter(prefix="/predict", tags=["Predict"])

@router.post("", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed")

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = run_prediction(image, file.filename)
    return result