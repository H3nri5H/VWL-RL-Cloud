# Backend Dockerfile - FastAPI + Ray RLlib Inference
# Zustandsbehaftet: Model wird beim Start geladen und bleibt im RAM
# BUILD CONTEXT: Project Root (not backend/)

FROM python:3.11-slim

# Metadata
LABEL maintainer="H3nri5H"
LABEL description="VWL-RL Backend: FastAPI Inference Service"

# Working Directory
WORKDIR /app

# System Dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY backend/ ./backend/
COPY envs/ ./envs/

# Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose Port (Cloud Run verwendet 8080)
EXPOSE 8080

# Start FastAPI Server
CMD ["uvicorn", "backend.serve:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
