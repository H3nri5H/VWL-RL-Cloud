FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install minimal dependencies for frontend
RUN pip install --no-cache-dir \
    streamlit==1.29.0 \
    plotly==5.18.0 \
    requests==2.31.0 \
    numpy==1.26.4

# Copy frontend code
COPY frontend/ .

EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
