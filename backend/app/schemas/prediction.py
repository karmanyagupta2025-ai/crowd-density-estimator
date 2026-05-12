from pydantic import BaseModel

class PredictionResponse(BaseModel):
    filename: str
    width: int
    height: int
    predicted_count: int
    density_level: str