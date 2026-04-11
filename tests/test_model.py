import pytest
import torch
from app.model_handler import GenrePredictor

def test_predictor_initialization():
    """
    Verifies that the predictor can initialize and fetch 
    artifacts from Hugging Face without crashing.
    """
    device = torch.device("cpu") # Use CPU for Jenkins environment
    repo_id = "silvermonkey/roberta-genre-classifier"
    
    try:
        predictor = GenrePredictor(repo_id=repo_id, device=device)
        assert predictor.model is not None
        assert len(predictor.genres) > 0
    except Exception as e:
        pytest.fail(f"Predictor failed to initialize: {e}")