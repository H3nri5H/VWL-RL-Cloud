# VWL-RL-Cloud 🎓

**Volkswirtschafts-Simulation mit Reinforcement Learning + Cloud Deployment**  
DHSH Module: Fortgeschrittene KI-Anwendungen & Cloud & Big Data | Januar 2026

[![Status](https://img.shields.io/badge/Status-Clean-brightgreen)]() [![Python](https://img.shields.io/badge/Python-3.11-blue)]() [![Cloud](https://img.shields.io/badge/Cloud-GCP-orange)]()

---

## 🎯 Projektübersicht

Cloud-basiertes Reinforcement Learning System für volkswirtschaftliche Simulationen. Das System demonstriert moderne Cloud-Architektur mit klar getrennten zustandslosen und zustandsbehafteten Komponenten.

### Kern-Features

- **Custom Gymnasium Environment**: Volkswirtschaftssimulation mit Firmen, Haushalten und Regierung
- **Reinforcement Learning**: PPO-Agent lernt optimale Wirtschaftspolitik
- **Cloud-Native Architektur**: Frontend (zustandslos) + Backend (zustandsbehaftet)
- **Modern Stack**: Streamlit, FastAPI, Google Cloud Run

---

## 🏛️ Architektur

```
User (Browser)
    ↓
Frontend (Streamlit) - ZUSTANDSLOS
    │ Cloud Run, 1GB RAM
    │ Jeder Request unabhängig
    │ Horizontal skalierbar
    ↓ HTTP POST /simulate
Backend (FastAPI) - ZUSTANDSBEHAFTET
    │ Cloud Run, 2GB RAM
    │ Environment + Model im RAM
    │ Lazy Loading
    ↓
Economy Environment (Gymnasium)
    │ 10 Firmen (produzieren, setzen Preise)
    │ 50 Haushalte (konsumieren, sparen)
    │ 1 Regierung (Steuern, Ausgaben, Zinsen)
```

### Zustandstrennung

**Frontend (Zustandslos):**
- Streamlit Web UI
- Keine persistenten Daten zwischen Requests
- Kann beliebig viele Instanzen starten
- Load Balancing trivial

**Backend (Zustandsbehaftet):**
- Environment-Instanz im RAM
- Optional: RL-Model gecacht
- State bleibt zwischen Requests erhalten
- Simulation läuft serverseitig

---

## 🚀 Quick Start

### Lokale Entwicklung

```bash
# 1. Repo klonen
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud

# 2. Virtual Environment
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Dependencies installieren
pip install --upgrade pip
pip install -r requirements.txt

# 4. Backend starten (Terminal 1)
cd backend
python serve.py
# Backend läuft auf http://localhost:8080

# 5. Frontend starten (Terminal 2)
export BACKEND_URL=http://localhost:8080  # Linux/Mac
set BACKEND_URL=http://localhost:8080     # Windows
streamlit run frontend/app.py
# Frontend öffnet sich im Browser
```

### Cloud Deployment

```bash
# Backend deployen
gcloud builds submit --config=backend/cloudbuild.yaml
gcloud run deploy vwl-rl-backend \
  --image gcr.io/PROJECT_ID/vwl-rl-backend \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2

# Frontend deployen
gcloud builds submit --config=frontend/cloudbuild.yaml
gcloud run deploy vwl-rl-frontend \
  --image gcr.io/PROJECT_ID/vwl-rl-frontend \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 \
  --set-env-vars BACKEND_URL=https://vwl-rl-backend-XXX.run.app
```

---

## 📦 Projektstruktur

```
VWL-RL-Cloud/
├── backend/
│   ├── serve.py              # FastAPI Server mit /simulate Endpoint
│   ├── Dockerfile            # Backend Container
│   └── cloudbuild.yaml       # Build Config
├── frontend/
│   ├── app.py                # Streamlit UI (KEIN Mock mehr!)
│   ├── Dockerfile            # Frontend Container
│   └── cloudbuild.yaml       # Build Config
├── envs/
│   ├── __init__.py
│   └── economy_env.py        # Gymnasium Environment
├── train/
│   ├── __init__.py
│   └── train_single.py       # RL Training Scripts
├── tests/
│   ├── test_env.py           # Environment Tests
│   └── test_scenarios.py     # Szenario Tests
├── requirements.txt          # Python Dependencies
└── README.md                 # Diese Datei
```

---

## 📊 Environment Details

### EconomyEnv (Gymnasium.Env)

**Observation Space (5 Dimensionen):**
- BIP (normalisiert)
- Inflationsrate (-50% bis +50%)
- Arbeitslosenquote (0 bis 1)
- Staatsschulden (normalisiert)
- Zinssatz (0 bis 20%)

**Action Space (3 Dimensionen):**
- Steuersatz (0-50%)
- Staatsausgaben (0-1000 EUR)
- Zinssatz (0-20%)

**Reward-Funktion:**
```python
reward = (
    + bip_wachstum * 10.0
    - arbeitslosigkeit * 20.0
    - abs(inflation) * 15.0
    - abs(defizit) * 0.01
)
```

### Simulation Flow

1. Environment Reset (Startzustand)
2. Für jeden Step (1 Tag):
   - Firmen produzieren und setzen Preise
   - Haushalte konsumieren
   - Markt findet Gleichgewicht
   - Regierung setzt Policy (Action)
   - Makro-Variablen werden berechnet
3. Daten werden gesammelt und zurückgegeben

---

## 🧠 Backend API

### Endpoints

#### `GET /health`
```json
{
  "status": "healthy",
  "env_available": true,
  "model_loaded": false
}
```

#### `POST /simulate`
**Request:**
```json
{
  "environment": "FullEconomy-v0",
  "num_steps": 100,
  "scenario": "Normal",
  "use_rl_agent": false,
  "manual_params": {
    "tax_rate": 0.3,
    "gov_spending": 500.0,
    "interest_rate": 0.05
  }
}
```

**Response:**
```json
{
  "steps": [
    {
      "step": 0,
      "bip": 5000.0,
      "inflation": 0.02,
      "unemployment": 0.05,
      "debt": 1000.0,
      "tax_rate": 0.3,
      "gov_spending": 500.0,
      "interest_rate": 0.05
    },
    ...
  ],
  "summary": {
    "final_bip": 5300.0,
    "bip_growth": 6.0,
    "avg_inflation": 2.1,
    "avg_unemployment": 5.2,
    "final_debt": 1100.0
  }
}
```

---

## 🛠️ Entwicklung

### Tests ausführen

```bash
# Environment Tests
python tests/test_env.py

# Szenario Tests
python tests/test_scenarios.py
```

### Lokales Backend testen

```bash
# Terminal 1: Backend starten
cd backend
python serve.py

# Terminal 2: Curl Tests
curl http://localhost:8080/health

curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "environment": "FullEconomy-v0",
    "num_steps": 10,
    "scenario": "Normal",
    "use_rl_agent": false,
    "manual_params": {"tax_rate": 0.3, "gov_spending": 500, "interest_rate": 0.05}
  }'
```

---

## 🎯 Modul-Anforderungen Erfüllt

### Fortgeschrittene KI-Anwendungen
- ✅ Reinforcement Learning (PPO)
- ✅ Custom Gymnasium Environment
- ✅ Multi-Agent System (Firmen, Haushalte, Regierung)
- ✅ Reward Shaping & Normalisierung

### Cloud & Big Data
- ✅ **Zustandslose Komponente**: Frontend (Streamlit auf Cloud Run)
- ✅ **Zustandsbehaftete Komponente**: Backend (Environment + Model im RAM)
- ✅ **Cloud Deployment**: Google Cloud Run
- ✅ **Containerization**: Docker Images in GCR
- ✅ **CI/CD**: Cloud Build Pipelines

---

## 📝 Changelog

### v2.0 (27.01.2026) - Clean Refactor
- ✅ Backend: `/simulate` Endpoint mit vollständiger Environment-Integration
- ✅ Frontend: Mock-Daten entfernt, echte Backend-Kommunikation
- ✅ Architektur: Klare Zustandstrennung dokumentiert
- ✅ Code Quality: "Ball of Mud" eliminiert
- ✅ Testing: Backend Health Check im Frontend

### v1.0 (21.01.2026)
- ✅ Initial Release mit Mock-Daten
- ✅ Cloud Deployment Setup

---

## 👤 Autor

**H3nri5H** (Foxyy)  
DHSH - Fortgeschrittene KI-Anwendungen & Cloud & Big Data  
Januar 2026

---

## 📚 Weitere Dokumentation

- [DEVELOPMENT.md](DEVELOPMENT.md) - Detaillierte Entwicklungs-Anleitung
- [Backend API Docs](https://vwl-rl-backend-698656921826.europe-west1.run.app/docs) - FastAPI Swagger UI

---

**Status**: 🟢 **Clean & Production Ready** - Keine Mock-Daten mehr!
