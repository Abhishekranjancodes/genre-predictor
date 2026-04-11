import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pickle

class RoBERTaClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # Using AutoModel to match your training cell 3
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.drop = nn.Dropout(0.3)
        self.clf = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Taking the CLS token (first token) as per your training logic
        cls = out.last_hidden_state[:, 0, :]
        return self.clf(self.drop(cls))

class GenrePredictor:
    def __init__(self, model_path, mlb_path, device):
        self.device = device
        
        # Load MultiLabelBinarizer
        with open(mlb_path, "rb") as f:
            self.mlb = pickle.load(f)
        self.genres = list(self.mlb.classes_)
        
        # Initialize Model
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = RoBERTaClassifier(num_labels=len(self.genres))
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()

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

        # Decode based on your thresholding logic
        binary_preds = (probs > threshold).astype(int)
        predicted_labels = [self.genres[i] for i, v in enumerate(binary_preds) if v == 1]
        
        return {
            "genres": predicted_labels,
            "probabilities": {self.genres[i]: float(probs[i]) for i in range(len(self.genres))}
        }