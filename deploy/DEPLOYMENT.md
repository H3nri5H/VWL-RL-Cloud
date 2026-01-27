# ☁️ GCP Deployment Guide - VWL-RL-Cloud

**Domain:** `clubpilot.one` (IONOS)  
**Cloud:** Google Cloud Platform  
**Region:** europe-west3 (Frankfurt)

---

## 📄 **Voraussetzungen**

### 1. Google Cloud Account
- Free Trial: $300 Credits für 90 Tage
- Signup: https://cloud.google.com/free

### 2. gcloud CLI installieren
```bash
# Windows (PowerShell als Admin):
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe

# Linux/Mac:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authentifizieren:
gcloud auth login
```

### 3. GCP Project erstellen
```bash
# Project ID wählen (global eindeutig!):
export PROJECT_ID="vwl-rl-cloud-$(date +%s)"

# Project erstellen:
gcloud projects create $PROJECT_ID

# Project setzen:
gcloud config set project $PROJECT_ID

# Billing aktivieren (WICHTIG!):
# Gehe zu: https://console.cloud.google.com/billing
# Wähle: Project $PROJECT_ID
# Aktiviere: Billing Account
```

---

## 🚀 **Deployment**

### **Option 1: Automatisches Deployment** (EMPFOHLEN)

```bash
# 1. Repository klonen (falls noch nicht geschehen)
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud

# 2. Environment Variables setzen
export GCP_PROJECT_ID="deine-project-id"  # Von oben
export GCP_REGION="europe-west3"          # Frankfurt
export GCS_BUCKET="vwl-rl-models"         # Bucket Name

# 3. Deploy-Script ausführen
bash deploy/deploy.sh

# Das war's! Script macht alles automatisch:
#  - APIs aktivieren
#  - Storage Bucket erstellen
#  - Backend deployen (FastAPI)
#  - Frontend deployen (Streamlit)
#  - Training Job erstellen
```

**Dauer:** ~10-15 Minuten

---

### **Option 2: Manuelles Deployment**

#### **Schritt 1: APIs aktivieren**
```bash
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com
```

#### **Schritt 2: Storage Bucket**
```bash
gsutil mb -l europe-west3 gs://vwl-rl-models
```

#### **Schritt 3: Backend**
```bash
# Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/vwl-rl-backend backend/

# Deploy
gcloud run deploy vwl-rl-backend \
    --image gcr.io/$PROJECT_ID/vwl-rl-backend \
    --platform managed \
    --region europe-west3 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

#### **Schritt 4: Frontend**
```bash
# Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/vwl-rl-frontend frontend/

# Deploy (mit Backend URL)
BACKEND_URL=$(gcloud run services describe vwl-rl-backend \
    --region europe-west3 \
    --format 'value(status.url)')

gcloud run deploy vwl-rl-frontend \
    --image gcr.io/$PROJECT_ID/vwl-rl-frontend \
    --platform managed \
    --region europe-west3 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --set-env-vars "BACKEND_URL=$BACKEND_URL"
```

#### **Schritt 5: Training Job**
```bash
# Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/vwl-rl-training train/

# Job erstellen
gcloud run jobs create vwl-rl-training \
    --image gcr.io/$PROJECT_ID/vwl-rl-training \
    --region europe-west3 \
    --memory 4Gi \
    --cpu 4 \
    --task-timeout 3600 \
    --set-env-vars "NUM_ITERATIONS=100,MAX_YEARS=10,GCS_BUCKET=vwl-rl-models"
```

---

## 🌐 **Domain Setup: clubpilot.one**

### **Schritt 1: Domain Mapping in GCP**

```bash
# Frontend URL holen
FRONTEND_SERVICE="vwl-rl-frontend"
REGION="europe-west3"

# Domain Mapping erstellen
gcloud run domain-mappings create \
    --service=$FRONTEND_SERVICE \
    --domain=clubpilot.one \
    --region=$REGION

# DNS Records anzeigen (kopiere diese!)
gcloud run domain-mappings describe clubpilot.one \
    --region=$REGION
```

**Output:**
```
Resource Records:
  Type: CNAME
  Name: clubpilot.one
  RRData: ghs.googlehosted.com.
```

---

### **Schritt 2: DNS bei IONOS konfigurieren**

1. **Login bei IONOS:**
   - Gehe zu: https://www.ionos.de/
   - Login mit deinem Account
   - Navigation: **Domains** → **clubpilot.one**

2. **DNS-Einstellungen öffnen:**
   - Klicke auf **DNS**
   - Wähle **Expert Mode** oder **DNS Records verwalten**

3. **CNAME Record hinzufügen:**
   ```
   Type:  CNAME
   Name:  @                    (oder "clubpilot.one")
   Value: ghs.googlehosted.com.
   TTL:   3600                  (1 Stunde)
   ```

4. **OPTIONAL: www-Subdomain:**
   ```
   Type:  CNAME
   Name:  www
   Value: ghs.googlehosted.com.
   TTL:   3600
   ```

5. **Speichern**

---

### **Schritt 3: SSL-Zertifikat (automatisch)**

- Google Cloud Run erstellt **automatisch** ein SSL-Zertifikat
- **Dauer:** 15-60 Minuten nach DNS-Propagierung
- **Prüfen:**
  ```bash
  gcloud run domain-mappings describe clubpilot.one --region=europe-west3
  # Status: ACTIVE = Fertig!
  ```

---

### **DNS Propagierung prüfen:**

```bash
# Windows:
nslookup clubpilot.one

# Linux/Mac:
dig clubpilot.one

# Online Tool:
# https://dnschecker.org
```

**Erwartetes Ergebnis:**
```
clubpilot.one.  3600  IN  CNAME  ghs.googlehosted.com.
```

---

## 🏋️  **Training starten**

### **Cloud Run Job manuell starten:**
```bash
gcloud run jobs execute vwl-rl-training --region=europe-west3
```

### **Logs anschauen:**
```bash
gcloud logging read \
    "resource.type=cloud_run_job AND resource.labels.job_name=vwl-rl-training" \
    --limit 50 \
    --format json
```

### **Training-Fortschritt:**
- Dashboard: https://console.cloud.google.com/run/jobs
- Wähle: `vwl-rl-training`
- Tab: **Logs**

### **Trainierte Models:**
```bash
# Models im Bucket anzeigen:
gsutil ls -r gs://vwl-rl-models/models/

# Model herunterladen:
gsutil -m cp -r gs://vwl-rl-models/models/checkpoint_final ./checkpoints/
```

---

## 💰 **Kosten**

### **Monatliche Schätzung:**

| Service | Free Tier | Kosten |
|---------|-----------|--------|
| **Frontend** (Cloud Run) | 2 Mio Requests | 2-5€/Monat |
| **Backend** (Cloud Run) | 2 Mio Requests | 5-10€/Monat |
| **Storage** (GCS) | 5 GB | 0.50€/Monat |
| **Training Job** (1x/Woche) | - | 1-2€/Lauf |
| **Traffic** | 1 GB Egress | 0.50€/Monat |
| **GESAMT** | - | **~10-20€/Monat** |

**Free Tier Details:**
- Cloud Run: 2 Millionen Requests/Monat
- Cloud Storage: 5 GB
- Cloud Build: 120 Build-Minuten/Tag

**Tipp:** Bei wenig Traffic bleibt man oft im Free Tier!

---

## ⚙️ **Environment Variables**

### **Frontend:**
```bash
BACKEND_URL=https://vwl-rl-backend-xxx-ew.a.run.app
```

### **Backend:**
```bash
MODEL_PATH=/app/checkpoints/checkpoint_final
GCS_BUCKET=vwl-rl-models
```

### **Training:**
```bash
NUM_ITERATIONS=100
MAX_YEARS=10
GCS_BUCKET=vwl-rl-models
```

---

## 🔧 **Troubleshooting**

### **Problem: "Permission denied"**
```bash
# Service Account Permissions prüfen:
gcloud projects get-iam-policy $PROJECT_ID

# Storage Admin hinzufügen:
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
    --role="roles/storage.admin"
```

### **Problem: "Build failed"**
```bash
# Logs anschauen:
gcloud builds log --stream LATEST

# Quota prüfen:
gcloud compute project-info describe --project=$PROJECT_ID
```

### **Problem: "Domain not verified"**
1. Gehe zu: https://search.google.com/search-console
2. Add Property: `clubpilot.one`
3. Verify via DNS TXT Record (IONOS)

---

## 🔄 **Updates deployen**

```bash
# Code ändern
git pull

# Neu deployen (automatisch)
bash deploy/deploy.sh

# ODER manuell:
gcloud builds submit --tag gcr.io/$PROJECT_ID/vwl-rl-frontend frontend/
gcloud run deploy vwl-rl-frontend --image gcr.io/$PROJECT_ID/vwl-rl-frontend --region=europe-west3
```

---

## 📈 **Monitoring**

### **Cloud Console:**
- Dashboard: https://console.cloud.google.com/run
- Metrics: CPU, Memory, Requests, Latency
- Logs: Realtime & Historical

### **Uptime Monitoring:**
```bash
# Frontend Health Check:
curl https://clubpilot.one/_stcore/health

# Backend Health Check:
curl https://BACKEND_URL/health
```

---

## 🎯 **Production Checklist**

- [ ] GCP Billing aktiviert
- [ ] APIs enabled
- [ ] Storage Bucket erstellt
- [ ] Backend deployed & healthy
- [ ] Frontend deployed & healthy
- [ ] Training Job getestet
- [ ] Domain Mapping erstellt
- [ ] DNS bei IONOS konfiguriert
- [ ] SSL-Zertifikat aktiv (15-60 Min)
- [ ] Custom Domain erreichbar: https://clubpilot.one
- [ ] Backend-Integration funktioniert
- [ ] Training läuft & speichert Models

---

## 📞 **Support**

**GCP Documentation:**
- Cloud Run: https://cloud.google.com/run/docs
- Cloud Storage: https://cloud.google.com/storage/docs
- Domain Mapping: https://cloud.google.com/run/docs/mapping-custom-domains

**IONOS DNS:**
- Help Center: https://www.ionos.de/hilfe/domains/
- DNS Settings: https://www.ionos.de/hilfe/domains/dns-einstellungen/

---

**Status:** 🟢 Ready for Production  
**Last Update:** 27.01.2026  
**Version:** 1.0.0
