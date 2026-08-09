"""
FastAPI app serving sentiment predictions from the fine-tuned DistilBERT model.
Run command: uvicorn src.app:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#MODEL_DIR = "models/sentiment-distilbert"
MODEL_DIR = "Srinija2/sentiment-distilbert"  

app = FastAPI(title="Sentiment Classifier API")

# Load model + tokenizer once at startup , for performanc sake
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # inference mode, disables dropout etc.

LABEL_MAP = {0: "negative", 1: "positive"}

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sentiment classifier API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
        return_token_type_ids=False,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)

    return PredictionResponse(
        text=request.text,
        label=LABEL_MAP[predicted_class.item()],
        confidence=round(confidence.item(), 4),
    )