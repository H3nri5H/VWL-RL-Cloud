# VWL-RL-Cloud 🎓

**Volkswirtschafts-Simulation mit Reinforcement Learning + Cloud-Native Architecture**  
DHSH Module: Fortgeschrittene KI-Anwendungen & Cloud & Big Data | Januar 2026

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]() [![Python](https://img.shields.io/badge/Python-3.11-blue)]() [![GCP](https://img.shields.io/badge/GCP-Cloud%20Native-orange)]() [![Kubernetes](https://img.shields.io/badge/Kubernetes-GKE-blue)]()

---

## 🏗️ **Cloud-Native Architecture**

```
┌─────────────────────────────────────────────────────┐
│  DEVELOPER (Lokal)                                  │
│  - Code schreiben, testen                           │
│  - git push → triggert Cloud Build                  │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  CLOUD BUILD (CI/CD) ✅                             │
│  - Run Tests                                        │
│  - Build Docker Images                              │
│  - Push to GCR                                      │
│  - Auto-Deploy to GKE                               │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  GOOGLE KUBERNETES ENGINE (GKE) ✅                  │
│  ┌─────────────────┐  ┌──────────────────┐         │
│  │ Frontend Pods   │  │ Backend Pods     │         │
│  │ (Streamlit)     │  │ (FastAPI+RL)     │         │
│  │ 3 Replicas      │  │ 2 Replicas       │         │
│  └─────────────────┘  └────────┬─────────┘         │
│                                  │                   │
│                                  │ Load Models       │
│                                  ▼                   │
│                     ┌────────────────────────┐      │
│                     │ CLOUD STORAGE (GCS) ✅ │      │
│                     │ - ppo_v1_10M.zip       │      │
│                     │ - ppo_v2_50M.zip       │      │
│                     └────────────────────────┘      │
└─────────────────────────────────────────────────────┘
              ▲
              │ Training Complete Event
              │
┌─────────────────────────────────────────────────────┐
│  PUB/SUB ✅                                         │
│  - training-events topic                            │
│  - backend subscribes → auto-loads new models       │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (Cloud Run Job) ✅               │
│  - Trains PPO model (24h, 10M steps)                │
│  - Uploads to GCS                                   │
│  - Publishes Pub/Sub event                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  TERRAFORM (Infrastructure as Code) ✅              │
│  - Defines all GCP resources                        │
│  - terraform apply → creates everything             │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 **Quick Start**

### **1. Lokale Entwicklung**

```bash
# Clone
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud

# Setup (Windows)
setup.bat

# OR Python Setup
python setup.py

# Test lokal
streamlit run frontend/app.py
```

### **2. Cloud Deployment**

#### **Schritt 1: Terraform Infrastructure**

```bash
cd terraform

# Config anpassen
cp terraform.tfvars.example terraform.tfvars
# Edit: project_id eintragen

# Deploy!
terraform init
terraform plan
terraform apply
```

**Erstellt:**
- ✅ GKE Cluster (2 Nodes, auto-scaling 1-5)
- ✅ Cloud Storage Buckets (models + logs)
- ✅ Pub/Sub Topics & Subscriptions

#### **Schritt 2: Build & Deploy**

```bash
# Setup Cloud Build Trigger (einmalig)
gcloud builds triggers create github \
  --repo-name=VWL-RL-Cloud \
  --repo-owner=H3nri5H \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml

# Oder manuell bauen:
gcloud builds submit --config=cloudbuild.yaml
```

**Cloud Build macht automatisch:**
1. Tests ausführen
2. Docker Images bauen (Backend + Frontend)
3. Push zu GCR
4. Deploy zu GKE

#### **Schritt 3: Access Application**

```bash
# Get External IPs
kubectl get services

# Frontend: http://<FRONTEND-EXTERNAL-IP>
# Backend:  http://<BACKEND-EXTERNAL-IP>
```

---

## 🏋️ **Training in der Cloud**

```bash
# Build Training Image
gcloud builds submit --config=train/cloudbuild-training.yaml

# Job läuft automatisch (24h)
# Check Status:
gcloud run jobs executions list

# Logs:
gcloud run jobs logs read rl-training-job
```

**Was passiert:**
1. Training läuft (10M steps, ~24h)
2. Model wird zu GCS hochgeladen
3. Pub/Sub Event wird publiziert
4. Backend lädt neues Model automatisch
5. Frontend kann neue Model-Version wählen

---

## 📊 **Projekt-Struktur**

```
VWL-RL-Cloud/
├── terraform/                  # Infrastructure as Code ✅
│   ├── main.tf                # Terraform Config
│   ├── gke.tf                 # Kubernetes Cluster
│   ├── storage.tf             # Cloud Storage Buckets
│   └── pubsub.tf              # Event Topics
│
├── k8s/                        # Kubernetes Manifests ✅
│   ├── backend-deployment.yaml
│   ├── frontend-deployment.yaml
│   └── README.md
│
├── backend/                    # FastAPI Inference API
│   ├── serve.py               # API mit GCS + Pub/Sub ✅
│   ├── Dockerfile
│   └── cloudbuild.yaml
│
├── frontend/                   # Streamlit Web UI
│   ├── app.py                 # UI mit Backend Integration ✅
│   ├── Dockerfile
│   └── cloudbuild.yaml
│
├── train/                      # Training Pipeline ✅
│   ├── train_cloud.py         # Cloud Training Script
│   ├── Dockerfile.training
│   └── cloudbuild-training.yaml
│
├── envs/                       # RL Environments
│   └── economy_env.py         # Gymnasium Env
│
├── tests/                      # Tests
│   └── test_env.py
│
├── cloudbuild.yaml            # Main CI/CD Pipeline ✅
└── README.md                  # Diese Datei
```

---

## 🎯 **Features**

### **Cloud-Native Technologies:**

✅ **Kubernetes (GKE)** - Container Orchestration  
✅ **Cloud Storage** - ML Model Persistence  
✅ **Pub/Sub** - Event-Driven Architecture  
✅ **Terraform** - Infrastructure as Code  
✅ **Cloud Build** - CI/CD Pipeline  
✅ **Cloud Run Jobs** - Training Workloads  

### **Application Features:**

- 🧠 **Multi-Model Support** - Wähle zwischen verschiedenen RL-Models
- 📊 **Live Simulation** - Interaktive Wirtschafts-Simulation
- ⚙️ **Manual/Auto Mode** - Manuelle Steuerung oder RL-Agent
- 📈 **Real-time Visualisierung** - BIP, Inflation, Arbeitslosigkeit
- 🔄 **Auto-Scaling** - Horizontal Pod Autoscaler in GKE
- 🔐 **IAM Security** - Service Accounts mit Least Privilege

---

## 🎓 **Modul-Anforderungen**

### ✅ **Fortgeschrittene KI-Anwendungen**
- [x] Reinforcement Learning (PPO)
- [x] Custom Gymnasium Environment
- [x] Multi-Agent Simulation
- [x] Reward Shaping

### ✅ **Cloud & Big Data**
- [x] **Zustandslos**: Frontend (Streamlit)
- [x] **Zustandsbehaftet**: Backend (Model in RAM)
- [x] **Kubernetes**: GKE Deployment
- [x] **Cloud Storage**: GCS für Models
- [x] **Pub/Sub**: Event-Driven
- [x] **Terraform**: IaC
- [x] **CI/CD**: Cloud Build

---

## 🔧 **Development Workflow**

### **Lokal entwickeln:**

```bash
# Code ändern
vim backend/serve.py

# Lokal testen
python tests/test_env.py
streamlit run frontend/app.py

# Commit
git add .
git commit -m "Feature: XYZ"
git push origin main
```

### **Cloud Build triggert automatisch:**
- ✅ Tests
- ✅ Build
- ✅ Deploy

### **Release erstellen:**

```bash
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0
```

---

## 📚 **Dokumentation**

- [Terraform Guide](terraform/README.md)
- [Kubernetes Guide](k8s/README.md)
- [Training Guide](train/README.md)
- [Development Guide](DEVELOPMENT.md)

---

## 💰 **Kosten (Geschätzt)**

| Service | Nutzung | Kosten/Monat |
|---------|---------|-------------|
| GKE Cluster | 2 Nodes e2-standard-2 | ~€60 |
| Cloud Storage | 10GB Models | ~€0.20 |
| Pub/Sub | 100k Messages | Free Tier |
| Cloud Build | 120 Builds/Monat | Free Tier |
| Training Job | 1x/Woche (24h) | ~€40 |
| **TOTAL** | | **~€100/Monat** |

**Free Tier beachten:**
- Cloud Build: 120 Build-Minuten/Tag kostenlos
- Cloud Storage: 5GB kostenlos
- GKE: $74.40/Monat Cluster-Fee (1 Zonal Cluster)

---

## 👤 **Autor**

**H3nri5H** (Foxyy)  
DHSH - Fortgeschrittene KI-Anwendungen & Cloud & Big Data  
Januar 2026

---

## 📝 **Changelog**

### v2.0 (27.01.2026) - Cloud-Native Architecture
- ✅ Terraform Infrastructure as Code
- ✅ Kubernetes (GKE) Deployment
- ✅ Cloud Storage Integration
- ✅ Pub/Sub Event System
- ✅ Cloud Build CI/CD Pipeline
- ✅ Training Jobs in Cloud
- ✅ Multi-Model Support

### v1.0 (21.01.2026) - Initial Release
- ✅ Economy Environment
- ✅ Streamlit Frontend
- ✅ FastAPI Backend
- ✅ Cloud Run Deployment

---

**Status**: 🟢 **Production Ready (Cloud-Native v2.0)**

🚀 **Full Stack:** Local Development → Git Push → Auto Build → Auto Deploy → Live!
