# Training Dockerfile - Ray RLlib Training in Cloud
# Für Cloud Run Jobs: Training läuft, speichert Model, beendet sich

FROM python:3.11-slim

# Metadata
LABEL maintainer="H3nri5H"
LABEL description="VWL-RL Training: Ray RLlib Cloud Training"

# Working Directory
WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir google-cloud-storage

# Copy Application Code
COPY train/ ./train/
COPY envs/ ./envs/

# Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONWARNINGS=ignore

# Training Config (kann via ENV überschrieben werden)
ENV NUM_ITERATIONS=100
ENV MAX_YEARS=10
ENV GCS_BUCKET=""

# Entrypoint: Training Script
ENTRYPOINT ["python", "train/train_cloud.py"]
