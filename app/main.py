from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from .model_handler import GenrePredictor

app = FastAPI(title="RoBERTa Genre Classifier")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = GenrePredictor(
    model_path="artifacts/roberta_base_best.pt",
    mlb_path="artifacts/mlb.pkl",
    device=DEVICE
)

class TextInput(BaseModel):
    text: str
    threshold: float = 0.75

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(DEVICE)}

@app.post("/predict")
async def get_prediction(payload: TextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = predictor.predict(payload.text, threshold=payload.threshold)
    return result