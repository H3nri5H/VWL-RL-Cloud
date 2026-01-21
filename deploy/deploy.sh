#!/bin/bash
# Cloud Deployment Script für GCP Cloud Run

set -e

echo "🚀 VWL-RL-Cloud Deployment"
echo "============================"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
else
    echo "⚠️  .env file not found! Using .env.example as template"
    cp .env.example .env
    echo "❗ Please edit .env with your GCP credentials"
    exit 1
fi

# Check GCP CLI
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ GCP Project: $GCP_PROJECT_ID"
echo "✅ Region: $GCP_REGION"

# Set GCP project
gcloud config set project $GCP_PROJECT_ID

echo ""
echo "📦 Building Docker Images..."

# Build Frontend
echo "Building Frontend..."
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/vwl-frontend \
    --dockerfile=frontend/Dockerfile .

# Build Backend
echo "Building Backend..."
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/vwl-backend \
    --dockerfile=backend/Dockerfile .

echo ""
echo "☁️  Deploying to Cloud Run..."

# Deploy Frontend (Zustandslos)
echo "Deploying Frontend (stateless)..."
gcloud run deploy vwl-frontend \
    --image gcr.io/$GCP_PROJECT_ID/vwl-frontend \
    --platform managed \
    --region $GCP_REGION \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 10 \
    --set-env-vars "BACKEND_URL=$BACKEND_URL"

# Deploy Backend (Zustandsbehaftet - mit Model)
echo "Deploying Backend (stateful)..."
gcloud run deploy vwl-backend \
    --image gcr.io/$GCP_PROJECT_ID/vwl-backend \
    --platform managed \
    --region $GCP_REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 5 \
    --timeout 300 \
    --set-env-vars "MODEL_PATH=$MODEL_PATH,GCS_BUCKET=$GCS_BUCKET"

echo ""
echo "✅ Deployment Complete!"
echo "============================"
echo "Frontend URL: $(gcloud run services describe vwl-frontend --region $GCP_REGION --format 'value(status.url)')"
echo "Backend URL: $(gcloud run services describe vwl-backend --region $GCP_REGION --format 'value(status.url)')"
echo ""
echo "📝 Update .env with these URLs for local development"
