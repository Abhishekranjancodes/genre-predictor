import os
import time
import logging
import json
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .model_handler import GenrePredictor

# Structured JSON Logger
class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for Logstash ingestion."""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": "roberta-backend",
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields attached to the record
        for key in ("endpoint", "method", "status_code", "duration_ms",
                     "text_length", "threshold", "predicted_genres",
                     "client_ip"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value
        return json.dumps(log_entry)


logger = logging.getLogger("roberta-backend")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.handlers = [handler]

# FastAPI Application
app = FastAPI(title="Movie/Book Genre Classifier")

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
    logger.info("Model loaded successfully", extra={
        "endpoint": "startup",
        "status_code": 200,
    })
except Exception as e:
    logger.error(f"Error initializing model: {e}", extra={
        "endpoint": "startup",
        "status_code": 503,
    })
    predictor = None

class TextInput(BaseModel):
    text: str
    threshold: float = 0.75

# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)

    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code}",
        extra={
            "endpoint": str(request.url.path),
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": duration,
            "client_ip": request.client.host if request.client else "unknown",
        },
    )
    return response

# Endpoints
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

    logger.info(
        f"Prediction completed — {len(result['genres'])} genre(s) found",
        extra={
            "endpoint": "/predict",
            "method": "POST",
            "text_length": len(payload.text),
            "threshold": payload.threshold,
            "predicted_genres": result["genres"],
            "status_code": 200,
        },
    )
    return result