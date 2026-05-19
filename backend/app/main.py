from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import io
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
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        count = 0  # replace with real count from model
        result_img = img  # replace with heatmap image from model
        # =====================================================

        buffer = io.BytesIO()
        result_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse({
            "count": count,
            "heatmap_base64": f"data:image/png;base64,{encoded}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")