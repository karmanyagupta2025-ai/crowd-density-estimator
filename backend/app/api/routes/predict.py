from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64
import io

from app.services.predictor import run_prediction

router = APIRouter(prefix="/predict", tags=["Predict"])

@router.post("")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, and WEBP images are allowed")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received")

    try:
        result_img, count = run_prediction(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    buffer = io.BytesIO()
    result_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return JSONResponse({
        "count": count,
        "heatmap_base64": f"data:image/png;base64,{encoded}"
    })