FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching — deps rarely change vs code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and trained model
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]