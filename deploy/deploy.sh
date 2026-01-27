#!/bin/bash
# VWL-RL Cloud Deployment Script
# Deployt alle Services zu Google Cloud Platform

set -e  # Exit bei Fehler

# === KONFIGURATION ===
PROJECT_ID="${GCP_PROJECT_ID:-vwl-rl-cloud}"
REGION="${GCP_REGION:-europe-west3}"  # Frankfurt
BUCKET_NAME="${GCS_BUCKET:-vwl-rl-models}"

# Service Namen
BACKEND_SERVICE="vwl-rl-backend"
FRONTEND_SERVICE="vwl-rl-frontend"
TRAINING_JOB="vwl-rl-training"

echo "==============================================="
echo "☁️  VWL-RL Cloud Deployment"
echo "==============================================="
echo "Project: $PROJECT_ID"
echo "Region:  $REGION"
echo "Bucket:  $BUCKET_NAME"
echo "==============================================="
echo ""

# === 1. GCP PROJECT SETUP ===
echo "🔧 1/6: GCP Project Setup..."

# Project setzen
gcloud config set project $PROJECT_ID

# APIs aktivieren
echo "  Aktiviere APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    --quiet

echo "  ✅ APIs aktiviert"

# === 2. CLOUD STORAGE BUCKET ===
echo "💾 2/6: Cloud Storage Bucket..."

if gsutil ls -b gs://$BUCKET_NAME 2>/dev/null; then
    echo "  Bucket existiert bereits: gs://$BUCKET_NAME"
else
    echo "  Erstelle Bucket: gs://$BUCKET_NAME"
    gsutil mb -l $REGION gs://$BUCKET_NAME
    echo "  ✅ Bucket erstellt"
fi

# === 3. BACKEND DEPLOYMENT ===
echo "🚀 3/6: Backend Service (FastAPI)..."

echo "  Building Backend Image..."
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$BACKEND_SERVICE \
    --project $PROJECT_ID \
    --quiet \
    backend/

echo "  Deploying Backend to Cloud Run..."
gcloud run deploy $BACKEND_SERVICE \
    --image gcr.io/$PROJECT_ID/$BACKEND_SERVICE \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --set-env-vars "MODEL_PATH=/app/checkpoints/checkpoint_final,GCS_BUCKET=$BUCKET_NAME" \
    --project $PROJECT_ID \
    --quiet

# Backend URL speichern
BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE \
    --region $REGION \
    --format 'value(status.url)' \
    --project $PROJECT_ID)

echo "  ✅ Backend deployed: $BACKEND_URL"

# === 4. FRONTEND DEPLOYMENT ===
echo "🎨 4/6: Frontend Service (Streamlit)..."

echo "  Building Frontend Image..."
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$FRONTEND_SERVICE \
    --project $PROJECT_ID \
    --quiet \
    frontend/

echo "  Deploying Frontend to Cloud Run..."
gcloud run deploy $FRONTEND_SERVICE \
    --image gcr.io/$PROJECT_ID/$FRONTEND_SERVICE \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --set-env-vars "BACKEND_URL=$BACKEND_URL" \
    --project $PROJECT_ID \
    --quiet

# Frontend URL speichern
FRONTEND_URL=$(gcloud run services describe $FRONTEND_SERVICE \
    --region $REGION \
    --format 'value(status.url)' \
    --project $PROJECT_ID)

echo "  ✅ Frontend deployed: $FRONTEND_URL"

# === 5. TRAINING JOB ===
echo "🏋️  5/6: Training Job..."

echo "  Building Training Image..."
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$TRAINING_JOB \
    --project $PROJECT_ID \
    --quiet \
    train/

echo "  Creating Cloud Run Job..."
gcloud run jobs create $TRAINING_JOB \
    --image gcr.io/$PROJECT_ID/$TRAINING_JOB \
    --region $REGION \
    --memory 4Gi \
    --cpu 4 \
    --task-timeout 3600 \
    --set-env-vars "NUM_ITERATIONS=100,MAX_YEARS=10,GCS_BUCKET=$BUCKET_NAME" \
    --project $PROJECT_ID \
    --quiet 2>/dev/null || \
gcloud run jobs update $TRAINING_JOB \
    --image gcr.io/$PROJECT_ID/$TRAINING_JOB \
    --region $REGION \
    --memory 4Gi \
    --cpu 4 \
    --task-timeout 3600 \
    --set-env-vars "NUM_ITERATIONS=100,MAX_YEARS=10,GCS_BUCKET=$BUCKET_NAME" \
    --project $PROJECT_ID \
    --quiet

echo "  ✅ Training Job erstellt"

# === 6. CUSTOM DOMAIN (Optional) ===
echo "🌐 6/6: Custom Domain Setup..."
echo "  ⚠️  Manueller Schritt erforderlich!"
echo ""
echo "  1. Domain Mapping erstellen:"
echo "     gcloud run domain-mappings create --service=$FRONTEND_SERVICE --domain=clubpilot.one --region=$REGION"
echo ""
echo "  2. DNS bei IONOS konfigurieren:"
echo "     - Gehe zu IONOS DNS Management"
echo "     - Füge CNAME Record hinzu:"
echo "       Name: @ (oder www)"
echo "       Wert: ghs.googlehosted.com"
echo ""
echo "  3. SSL wird automatisch von Google bereitgestellt (kann 15 Min dauern)"
echo ""

# === DEPLOYMENT SUMMARY ===
echo "==============================================="
echo "✅ DEPLOYMENT ERFOLGREICH!"
echo "==============================================="
echo ""
echo "🎨 Frontend:  $FRONTEND_URL"
echo "🚀 Backend:   $BACKEND_URL"
echo "💾 Storage:   gs://$BUCKET_NAME"
echo ""
echo "==============================================="
echo "🏋️  Training starten:"
echo "==============================================="
echo "gcloud run jobs execute $TRAINING_JOB --region=$REGION"
echo ""
echo "==============================================="
echo "🌐 Custom Domain (clubpilot.one):"
echo "==============================================="
echo "1. Domain Mapping:"
echo "   gcloud run domain-mappings create \\"
echo "     --service=$FRONTEND_SERVICE \\"
echo "     --domain=clubpilot.one \\"
echo "     --region=$REGION"
echo ""
echo "2. IONOS DNS:"
echo "   CNAME @ -> ghs.googlehosted.com"
echo ""
echo "==============================================="
echo "💰 Kosten-Schätzung:"
echo "==============================================="
echo "Frontend:  ~2-5€/Monat (Zustandslos, schnell)"
echo "Backend:   ~5-10€/Monat (Zustandsbehaftet, Model im RAM)"
echo "Storage:   ~0.50€/Monat (Checkpoints)"
echo "Training:  ~1-2€/Lauf (1h Runtime)"
echo ""
echo "Free Tier: 2 Mio Requests/Monat kostenlos"
echo "==============================================="
echo ""
