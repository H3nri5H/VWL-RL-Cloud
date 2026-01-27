# Frontend Dockerfile - Streamlit Web UI
# Zustandslos: Jeder Request ist unabhängig

FROM python:3.11-slim

# Metadata
LABEL maintainer="H3nri5H"
LABEL description="VWL-RL Frontend: Streamlit Web UI"

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
COPY frontend/ ./frontend/
COPY envs/ ./envs/

# Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Streamlit Config
RUN mkdir -p ~/.streamlit && \
    echo "[server]" > ~/.streamlit/config.toml && \
    echo "port = 8080" >> ~/.streamlit/config.toml && \
    echo "address = '0.0.0.0'" >> ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "enableCORS = false" >> ~/.streamlit/config.toml && \
    echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Expose Port
EXPOSE 8080

# Start Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
