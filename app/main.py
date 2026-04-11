import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .model_handler import GenrePredictor

app = FastAPI(title="RoBERTa Genre Classifier (MLOps Edition)")

# Enabled CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "silvermonkey/roberta-genre-classifier" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trigger the Hugging Face download on startup
try:
    predictor = GenrePredictor(repo_id=REPO_ID, device=DEVICE)
except Exception as e:
    print(f"Error initializing model: {e}")
    predictor = None

class TextInput(BaseModel):
    text: str
    threshold: float = 0.75

@app.get("/health")
def health_check():
    """Mandatory health check for Kubernetes liveness probes """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": str(DEVICE), "repo": REPO_ID}

@app.post("/predict")
async def get_prediction(payload: TextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    result = predictor.predict(payload.text, threshold=payload.threshold)
    return result