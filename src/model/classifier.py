"""Thin wrapper around the pretrained DistilBERT deception classifier.

The classifier is used strictly as a black box: the attack pipeline only ever
needs a predicted label and the model's confidence in that label.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DEVICE, MODEL_PATH


class DeceptionClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, text):
        """Return (predicted_label, confidence_in_that_label)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()
        return label, confidence
