FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install backend dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    pydantic==2.5.0 \
    numpy==1.26.4 \
    ray[rllib]==2.10.0 \
    gymnasium==0.29.1 \
    torch==2.1.0

# Copy backend + environment code
COPY backend/ ./backend/
COPY envs/ ./envs/

# Copy trained models (will be mounted from Cloud Storage in production)
COPY models/ ./models/ 

EXPOSE 8000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Run FastAPI
CMD ["uvicorn", "backend.serve:app", "--host", "0.0.0.0", "--port", "8000"]
