import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pickle
import os
from huggingface_hub import hf_hub_download

class RoBERTaClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.drop = nn.Dropout(0.3)
        self.clf = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.clf(self.drop(cls))

class GenrePredictor:
    def __init__(self, repo_id, device):
        self.device = device
        
        print(f"Fetching artifacts from Hugging Face: {repo_id}...")
        weights_path = hf_hub_download(repo_id=repo_id, filename="roberta_base_best.pt")
        mlb_path = hf_hub_download(repo_id=repo_id, filename="mlb.pkl")
        
        # Load MultiLabelBinarizer (Decoder)
        with open(mlb_path, "rb") as f:
            self.mlb = pickle.load(f)
        self.genres = list(self.mlb.classes_)
        
        # Initialize Model and Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = RoBERTaClassifier(num_labels=len(self.genres))
        
        # Load the downloaded weights
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.to(device).eval()
        print("Model and weights loaded successfully.")

    def predict(self, text, threshold=0.75):
        enc = self.tokenizer(
            text, max_length=512, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        binary_preds = (probs > threshold).astype(int)
        predicted_labels = [self.genres[i] for i, v in enumerate(binary_preds) if v == 1]
        
        return {
            "genres": predicted_labels,
            "probabilities": {self.genres[i]: float(probs[i]) for i in range(len(self.genres))}
        }