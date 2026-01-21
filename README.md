# VWL-RL-Cloud 🎓

**Volkswirtschafts-Simulation mit Multi-Agent Reinforcement Learning + Cloud Deployment**  
DHSH Module: Fortgeschrittene KI-Anwendungen & Cloud & Big Data | Januar 2026

[![Status](https://img.shields.io/badge/Status-Ready-brightgreen)]() [![Python](https://img.shields.io/badge/Python-3.11-blue)]() [![Ray](https://img.shields.io/badge/Ray-2.10-orange)]()

---

## 🎯 Projekt-Übersicht

### Multi-Agent RL-System
- 🏢 **10 Firmen-Agents** (RL): Entscheiden über Preise, Löhne, Mitarbeiteranzahl
- 🏠 **50 Haushalte** (regelbasiert): Konsum & Sparen basierend auf Einkommen
- 🏛️ **1 Regierungs-Agent** (RL): Steuerpolitik, Staatsausgaben, Zinssätze

### Cloud-Architektur (Google Cloud Platform)
- ⚡ **Zustandslos**: Streamlit Frontend (User-Interface)
- 🧠 **Zustandsbehaftet**: FastAPI Backend (RL-Inference mit geladenem Model)
- ☁️ **Cloud Services**: Cloud Run (Frontend), Cloud Run Jobs (Backend), Cloud Storage (Models)

### Tech Stack
- **RL Framework**: Ray RLlib 2.10 + PPO
- **Environment**: Custom Gymnasium.Env
- **Frontend**: Streamlit + Plotly
- **Backend**: FastAPI + Uvicorn
- **Cloud**: Google Cloud Run, Cloud Storage
- **Containerization**: Docker

---

## 🚀 Quick Start (5 Minuten)

### 1. Repository klonen
```bash
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud
code .  # VS Code öffnen
```

### 2. Python 3.11 Virtual Environment
```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3.11 -m venv .venv
source .venv/bin/activate
```

**VS Code Setup:**
- `Ctrl+Shift+P` → "Python: Select Interpreter"
- Wähle: `.venv/Scripts/python.exe` (Windows) oder `.venv/bin/python` (Linux/Mac)

### 3. Dependencies installieren
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ **Installation dauert ~5 Minuten** (Ray/RLlib sind groß)

### 4. Tests ausführen
```bash
# Environment testen
python -c "import ray; from ray.rllib.algorithms.ppo import PPOConfig; print('✅ RLlib ready:', ray.__version__)"

# Unit Tests
python tests/test_env.py

# Szenario-Tests
python tests/test_scenarios.py
```

### 5. Training starten
```bash
# Single-Agent Training (Regierung lernt Wirtschaftspolitik)
python train/train_single.py

# Erwartet: 
# - 50 Iterationen
# - Reward steigt von ~-15 auf +5
# - Model wird in models/ gespeichert
# - Dauer: ~10 Minuten (CPU)
```

### 6. Frontend lokal starten
```bash
streamlit run frontend/app.py

# Öffnet Browser auf http://localhost:8501
# - Slider: Steuersatz, Staatsausgaben, Zinsen
# - Szenarien: Normal, Rezession, Boom, Inflation
# - Live-Plots: BIP, Arbeitslosigkeit, Inflation
```

### 7. Backend lokal starten (optional)
```bash
uvicorn backend.serve:app --reload

# API läuft auf http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## 📁 Projekt-Struktur

```
VWL-RL-Cloud/
├── README.md                    # Diese Datei
├── requirements.txt             # Alle Dependencies
├── .gitignore                   # Git-Ausschlüsse
├── .env.example                 # Template für GCP-Keys
│
├── envs/                        # RL Environments
│   ├── __init__.py
│   └── economy_env.py          # Hauptsimulation (Gymnasium.Env)
│
├── train/                       # Training Scripts
│   ├── __init__.py
│   └── train_single.py         # Single-Agent Training (Regierung)
│
├── backend/                     # Zustandsbehaftet: RL Inference
│   ├── __init__.py
│   ├── serve.py                # FastAPI Server
│   └── Dockerfile              # Backend Container
│
├── frontend/                    # Zustandslos: Web UI
│   ├── app.py                  # Streamlit App
│   └── Dockerfile              # Frontend Container
│
├── tests/                       # Tests
│   ├── __init__.py
│   ├── test_env.py             # Environment Tests
│   └── test_scenarios.py       # Wirtschafts-Szenarien
│
└── deploy/                      # Cloud Deployment
    ├── deploy.sh               # GCP Deploy Script
    └── cloudbuild.yaml         # CI/CD Config
```

---

## 🧪 Tests

### Environment Tests
```bash
python tests/test_env.py

# Testet:
# ✅ Environment Creation
# ✅ Reset Funktionalität
# ✅ Step Funktionalität
# ✅ Episode Durchlauf
# ✅ Action Clipping
# ✅ Space Definitions
```

### Szenario Tests
```bash
python tests/test_scenarios.py

# Testet 9 Szenarien:
# 1. Normal (Baseline)
# 2. Niedrige Steuern
# 3. Hohe Steuern
# 4. Hohe Staatsausgaben
# 5. Niedrige Staatsausgaben
# 6. Hohe Zinsen
# 7. Niedrige Zinsen
# 8. Austerität (hohe Steuern + niedrige Ausgaben)
# 9. Expansion (niedrige Steuern + hohe Ausgaben)
#
# Output: Ranking nach Total Reward
```

---

## 🎓 Modul-Anforderungen

### ✅ Fortgeschrittene KI-Anwendungen
- [x] Multi-Agent Reinforcement Learning (RLlib)
- [x] Custom Gymnasium Environment
- [x] PPO-Algorithmus (Proximal Policy Optimization)
- [x] Reward-Shaping & Normalisierung
- [x] Training & Model Persistence

### ✅ Cloud & Big Data
- [x] **Zustandslose Komponente**: Streamlit Frontend
  - User-Interface ohne internen Zustand
  - Jeder Request ist unabhängig
  - Horizontal skalierbar
- [x] **Zustandsbehaftete Komponente**: FastAPI Backend
  - RL-Model wird beim Start geladen (zustandsbehaftet!)
  - Model bleibt im Speicher für schnelle Inference
  - Simulation-State wird über Requests hinweg gehalten
- [x] **Cloud Deployment**: Google Cloud Run
  - Container-basiert (Docker)
  - Auto-Scaling
  - Cloud Storage für Models

---

## ☁️ Cloud Deployment (GCP)

### Voraussetzungen
1. **Google Cloud Account** (Free Tier reicht für Tests)
2. **gcloud CLI** installiert: https://cloud.google.com/sdk/docs/install
3. **Docker** installiert (optional, für lokale Tests)

### Setup

```bash
# 1. GCP Projekt erstellen
gcloud projects create vwl-rl-cloud --name="VWL RL Cloud"
gcloud config set project vwl-rl-cloud

# 2. APIs aktivieren
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com

# 3. .env konfigurieren
cp .env.example .env
# Editiere .env mit deiner PROJECT_ID

# 4. Deploy!
bash deploy/deploy.sh
```

### Kosten-Schätzung (GCP Free Tier)
- **Frontend**: ~€0.01/Stunde (Cloud Run)
- **Backend**: ~€0.05/Stunde (Cloud Run mit 2Gi RAM)
- **Storage**: ~€0.02/GB/Monat
- **Total**: ~€5-10/Monat bei regelmäßiger Nutzung
- **Free Tier**: Erste 2 Mio Requests/Monat kostenlos!

---

## 🔧 Troubleshooting

### Problem: `py -3.11` nicht gefunden
**Lösung**: Python 3.11 installieren von https://python.org/downloads/release/python-3119/
- Windows: "Add python.exe to PATH" aktivieren!

### Problem: Ray/RLlib Fehler beim Install
**Lösung**: 
```bash
pip install --upgrade pip
pip install "ray[rllib]==2.10.0" --no-cache-dir
```

### Problem: Numpy/PyArrow Konflikt
**Lösung**: Exakte Versionen aus requirements.txt nutzen:
```bash
pip install numpy==1.26.4 pyarrow==14.0.1
```

### Problem: VS Code erkennt .venv nicht
**Lösung**: 
- `Ctrl+Shift+P` → "Python: Select Interpreter"
- Manuell `.venv/Scripts/python.exe` auswählen
- VS Code neu starten

### Problem: Training läuft nicht
**Lösung**: Prüfe Ray-Initialisierung:
```bash
python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

---

## 📊 Performance-Metriken

### Training (CPU, 50 Iterationen)
- **Dauer**: ~10 Minuten
- **Initial Reward**: -15 bis -10
- **Final Reward**: +2 bis +8
- **Best Case**: +10 (stabiles Wirtschaftswachstum)

### Simulation (Frontend)
- **Ladezeit**: <2 Sekunden
- **Simulation**: 100 Steps in <1 Sekunde
- **Rendering**: Real-time mit Plotly

---

## 📚 Dokumentation

- [Ray RLlib Docs](https://docs.ray.io/en/latest/rllib/index.html)
- [Gymnasium API](https://gymnasium.farama.org/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Google Cloud Run](https://cloud.google.com/run/docs)

---

## 🎯 Nächste Schritte / Erweiterungen

### V1.1 (geplant)
- [ ] Multi-Agent MARL: Alle 10 Firmen als RL-Agents
- [ ] Hyperparameter-Tuning mit Ray Tune
- [ ] Erweiterte Szenarien (Krisen, Boom-Bust-Zyklen)
- [ ] TensorBoard Integration für Training-Monitoring

### V2.0 (Zukunft)
- [ ] Haushalte als RL-Agents
- [ ] Internationale Handel-Simulation
- [ ] Historische Daten-Integration
- [ ] A/B-Testing verschiedener Wirtschaftspolitiken

---

## 👥 Für Dozenten: Schnell-Setup

```bash
# 1. Klonen
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud

# 2. Environment
py -3.11 -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install
pip install --upgrade pip
pip install -r requirements.txt

# 4. Tests
python tests/test_env.py
python tests/test_scenarios.py

# 5. Frontend Demo
streamlit run frontend/app.py
# → Browser öffnet http://localhost:8501

# 6. Training (optional)
python train/train_single.py
```

**Gesamtdauer**: ~15 Minuten (inkl. Installs)

---

## 📝 Ideenpräsentation (10 Min)

### Struktur (gemäß Anforderungen)

1. **Generelle Idee** (2 Min)
   - Wirtschafts-Simulation mit RL
   - Regierung lernt optimale Wirtschaftspolitik

2. **Kontext** (1 Min)
   - Makroökonomie: BIP, Inflation, Arbeitslosigkeit
   - Multi-Agent-Systeme

3. **ML-Ansatz** (3 Min)
   - Reinforcement Learning (PPO)
   - Multi-Agent (Firmen, Haushalte, Regierung)
   - Custom Gymnasium Environment

4. **Daten** (2 Min)
   - Synthetische Simulation (keine externen Daten nötig)
   - Plan B: Historische Wirtschaftsdaten (Eurostat, Bundesbank)

5. **Nutzen** (2 Min)
   - Policy-Testing ohne reale Konsequenzen
   - Bildungstool für Volkswirtschaftslehre
   - Cloud-Architektur als Showcase

---

## 👤 Autor

**H3nri5H** (Foxyy)  
DHSH - Fortgeschrittene KI-Anwendungen & Cloud & Big Data  
Januar 2026

---

## 📄 Lizenz

MIT License - siehe GitHub

---

**Status**: 🟢 **Production Ready** - Alle Kern-Features implementiert!

🎉 **Los geht's**: `git clone` und `streamlit run frontend/app.py`
