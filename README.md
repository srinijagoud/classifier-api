# Sentiment Classifier API

A fine-tuned DistilBERT model served as a production-style REST API, containerized with Docker and deployed on Railway. Built end-to-end: data prep → training → evaluation → serving → testing → deployment.

**Live demo**: [Railyway Link](https://classifier-api-production-5736.up.railway.app/docs)

## Overview
This project fine-tunes distilbert-base-uncased on the Rotten Tomatoes movie review dataset to classify text as positive or negative sentiment, then wraps it in a FastAPI service for real-time inference.

## Architecture
```
Data (Hugging Face Datasets)
        │
        ▼
Training (DistilBERT fine-tuning, GPU via Colab)
        │
        ▼
Model Registry (Hugging Face Hub)
        │
        ▼
Serving (FastAPI + PyTorch)
        │
        ▼
Container (Docker)
        │
        ▼
Deployment (Railway)
```

Model weights are hosted on the Hugging Face Hub rather than committed to this repo — the app downloads them at container startup via from_pretrained(). This keeps the git repo lightweight and follows standard model-registry practice.

## Results
Fine-tuned for 2 epochs on *~8,500* training examples:

| Metric | Score |
| --- | --- |
| Accuracy | 85.8% |
| F1 | 0.858 |
| Precision | 0.859 |
| Recall | 0.857 |

## Tech Stack
**Model**: DistilBERT (Hugging Face Transformers)

**Training**: PyTorch, Hugging Face Trainer, tracked with MLflow

**Serving**: FastAPI + Uvicorn

**Testing**: Pytest

**Containerization**: Docker

**Model registry**: Hugging Face Hub

**Deployment**: Railway

## API
**GET**

## Health check.
### Response:
```
json
{"status": "ok", "message": "Sentiment classifier API is running"}
```

**POST /predict**
Classify sentiment of input text.

#### Request:
```
json
{"text": "This movie was absolutely wonderful, I loved every minute."}
```

### Response:
```
json
{
  "text": "This movie was absolutely wonderful, I loved every minute.",
  "label": "positive",
  "confidence": 0.9883
}
```

## Running Locally
## Clone and set up environment
gitclone  https://github.com/srinijagoud/classifier-api.git

cd classifier-api

python -m venv venv

venv\Scripts\activate 

pip install -r requirements.txt

## Run the API
uvicorn src.app:app --reload

Visit *http://127.0.0.1:8000/docs* for the interactive Swagger UI.

## Running with Docker
docker build -t sentiment-classifier .
docker run -p 8000:8000 sentiment-classifier

## Running Tests
pytest tests/ -v

## Project Structure
```
classifier-api/
├── data/                   # train/val/test CSVs (not tracked in git)
├── src/
│   ├── data_prep.py        # loads and prepares the dataset
│   ├── train.py             # fine-tunes DistilBERT, logs to MLflow
│   └── app.py                # FastAPI serving layer
├── tests/
│   └── test_app.py         # API tests (health check, predictions, edge cases)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Known Limitations
- Binary classification only (positive/negative) — no neutral class, so ambiguous input is forced into one label, sometimes with low confidence (~0.5-0.6 range).
  
- Trained on movie review text specifically — may not generalize well to other domains (e.g. product reviews, social media, sarcasm).
  
- Max input length of 128 tokens — longer text is truncated.

## Future Improvements
- Add a confidence threshold to flag low-confidence predictions for human review
  
- Expand to a 3-class model (positive/neutral/negative)
  
- Add request logging and basic monitoring (e.g. prediction distribution over time)
  
- Set up CI/CD (GitHub Actions) to auto-run tests on push
