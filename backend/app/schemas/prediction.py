from pydantic import BaseModel

class PredictionResponse(BaseModel):
    count: int
    heatmap_base64: str