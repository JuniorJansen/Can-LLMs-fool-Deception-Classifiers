from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.config import MODEL_PATH, DEVICE


class DeceptionClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()

        return label, confidence