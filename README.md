# VWL-RL-Cloud 🎓

**Volkswirtschafts-Simulation mit Multi-Agent Reinforcement Learning + Cloud Deployment**  
DHSH Module: Fortgeschrittene KI-Anwendungen & Cloud & Big Data | Januar 2026

[![Status](https://img.shields.io/badge/Status-Ready-brightgreen)]() [![Python](https://img.shields.io/badge/Python-3.11-blue)]() [![Ray](https://img.shields.io/badge/Ray-2.10-orange)]()

---

## 🚀 **SUPER-QUICK START** (10 Minuten, KEINE Vorkenntnisse nötig!)

### ✅ **Option A: Windows Automatik-Setup** (EMPFOHLEN)

```cmd
# 1. Repo herunterladen
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud

# 2. Doppelklick auf setup.bat (ODER im Terminal:)
setup.bat

# Das war's! Script installiert alles automatisch:
#  - Prüft Python 3.11 (zeigt Download-Link falls fehlend)
#  - Erstellt venv
#  - Installiert alle Pakete
#  - Führt Tests aus
```

**Setup.bat macht automatisch:**
- ✅ Python 3.11 Check (mit Installations-Anleitung)
- ✅ Virtual Environment erstellen
- ✅ Pip upgrade
- ✅ Alle Dependencies installieren (~5 Min)
- ✅ Tests ausführen
- ✅ Bereit!

---

### ✅ **Option B: Python-Setup** (Alle Plattformen)

```bash
# 1. Repo klonen
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud

# 2. Automatisches Setup
python setup.py

# Alles wird automatisch gemacht!
```

---

### ✅ **Option C: Manuelles Setup** (Falls du es genau wissen willst)

#### **Schritt 1: Python 3.11 installieren** (falls nicht vorhanden)

**Windows:**
```cmd
# Prüfe ob Python 3.11 installiert:
py -3.11 --version

# Falls NICHT installiert:
# 1. Öffne: https://www.python.org/downloads/release/python-3119/
# 2. Download: "Windows installer (64-bit)"
# 3. Installiere mit "☑️ Add python.exe to PATH"

# ODER via winget (Windows 10/11):
winget install -e --id Python.Python.3.11
```

**Linux/Mac:**
```bash
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.11 python3.11-venv

# Mac (Homebrew):
brew install python@3.11
```

#### **Schritt 2: Repository klonen**
```bash
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud
```

#### **Schritt 3: Virtual Environment**
```bash
# Windows:
py -3.11 -m venv .venv
.venv\Scripts\activate

# Linux/Mac:
python3.11 -m venv .venv
source .venv/bin/activate
```

**VS Code Setup:**
- `Ctrl+Shift+P` → "Python: Select Interpreter"
- Wähle: `.venv/Scripts/python.exe` (Win) oder `.venv/bin/python` (Linux/Mac)

#### **Schritt 4: Dependencies installieren**
```bash
# Pip upgraden
python -m pip install --upgrade pip

# Alle Pakete installieren (~5 Minuten)
pip install -r requirements.txt
```

#### **Schritt 5: Tests**
```bash
# PYTHONPATH setzen (wichtig!)
# Windows:
set PYTHONPATH=.
# Linux/Mac:
export PYTHONPATH=.

# RLlib Test
python -c "import ray; from ray.rllib.algorithms.ppo import PPOConfig; print('✅ RLlib ready:', ray.__version__)"

# Environment Tests
python tests/test_env.py
```

---

## 🎉 **App starten**

```bash
# Frontend (Web-Interface)
streamlit run frontend/app.py
# → Öffnet http://localhost:8501

# Training (RL-Agent trainieren, ~10 Min)
python train/train_single.py

# Szenarien testen
python tests/test_scenarios.py
```

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

---

## 🔧 **Troubleshooting**

### ❌ **Problem: `py -3.11` nicht gefunden**

**Lösung:**
1. Python 3.11 installieren: https://python.org/downloads/release/python-3119/
2. **Wichtig**: ☑️ "Add python.exe to PATH" aktivieren!
3. Terminal **neu starten**
4. Test: `py -3.11 --version`

---

### ❌ **Problem: `ModuleNotFoundError: No module named 'envs'`**

**Lösung:**
```bash
# Windows:
set PYTHONPATH=.

# Linux/Mac:
export PYTHONPATH=.

# Dann nochmal:
python tests/test_env.py
```

**Permanent (VS Code):**
- Erstelle `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": ["."],
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}
```

---

### ❌ **Problem: Ray/RLlib Installation Fehler**

**Lösung:**
```bash
# Cache löschen und nochmal:
pip cache purge
pip install --no-cache-dir "ray[rllib]==2.10.0"

# Falls weiterhin Fehler:
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### ❌ **Problem: Gymnasium Version Conflict**

**Lösung:**
```bash
# Exakte Versionen erzwingen:
pip uninstall gymnasium ray -y
pip install gymnasium==0.28.1 "ray[rllib]==2.10.0"
```

**Grund**: Ray 2.10 braucht exakt Gymnasium 0.28.1 (bereits in requirements.txt gefixt)

---

### ❌ **Problem: VS Code erkennt venv nicht**

**Lösung:**
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Falls `.venv` nicht erscheint: "Enter interpreter path..."
3. Manuell auswählen:
   - Windows: `.venv\Scripts\python.exe`
   - Linux/Mac: `.venv/bin/python`
4. Terminal neu starten: `Ctrl+Shift+`` `

---

### ❌ **Problem: Streamlit startet nicht**

**Lösung:**
```bash
# Port 8501 belegt?
streamlit run frontend/app.py --server.port 8502

# Browser öffnet nicht automatisch?
streamlit run frontend/app.py --server.headless false
```

---

## 📁 Projekt-Struktur

```
VWL-RL-Cloud/
├── README.md                    # Diese Datei
├── requirements.txt             # Alle Dependencies (gefixt!)
├── setup.py                     # Automatisches Setup (neu!)
├── setup.bat                    # Windows One-Click Setup (neu!)
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
├── tests/                       # Tests (ohne pytest!)
│   ├── __init__.py
│   ├── test_env.py             # Environment Tests (gefixt!)
│   └── test_scenarios.py       # Wirtschafts-Szenarien (gefixt!)
│
└── deploy/                      # Cloud Deployment
    ├── deploy.sh               # GCP Deploy Script
    └── cloudbuild.yaml         # CI/CD Config
```

---

## 🎓 **Für Dozenten: Copy-Paste Setup**

```bash
# 1. Klonen
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud

# 2. Windows: Doppelklick setup.bat
#    Oder: python setup.py

# 3. Demo starten
streamlit run frontend/app.py
```

**Gesamtdauer**: ~10 Minuten (inkl. Downloads)

---

## 📊 Features im Frontend

- 🎲 **Interactive Sliders**: Steuersatz (0-50%), Staatsausgaben (0-1000€), Zinsen (0-20%)
- 🎬 **Szenarien**: Normal, Rezession, Boom, Inflation
- 📊 **Live-Plots**: BIP, Arbeitslosigkeit, Inflation (100 Steps)
- 🧠 **RL Toggle**: "RL-Agent nutzen" schaltet zwischen manuell/automatisch um
- 📊 **Metriken**: BIP-Wachstum, End-Werte, Durchschnitte

---

## ☁️ Cloud Deployment (Optional)

### Voraussetzungen
1. Google Cloud Account (Free Tier reicht)
2. gcloud CLI: https://cloud.google.com/sdk/docs/install

### Deploy
```bash
# .env konfigurieren
cp .env.example .env
# Edit: GCP_PROJECT_ID setzen

# Deploy!
bash deploy/deploy.sh
```

**Kosten**: ~€5-10/Monat (Free Tier: 2 Mio Requests kostenlos)

---

## 📚 Modul-Anforderungen

### ✅ Fortgeschrittene KI-Anwendungen
- [x] Multi-Agent Reinforcement Learning (Ray RLlib)
- [x] Custom Gymnasium Environment
- [x] PPO-Algorithmus
- [x] Reward-Shaping & Normalisierung

### ✅ Cloud & Big Data
- [x] **Zustandslose Komponente**: Streamlit Frontend
- [x] **Zustandsbehaftete Komponente**: FastAPI Backend (Model im RAM)
- [x] **Cloud Deployment**: Google Cloud Run
- [x] **Containerization**: Docker

---

## 📝 Ideenpräsentation (10 Min)

**Struktur (gemäß PDF-Anforderungen):**

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
   - Plan B: Historische Daten (Eurostat, Bundesbank)

5. **Nutzen** (2 Min)
   - Policy-Testing ohne reale Konsequenzen
   - Bildungstool
   - Cloud-Architektur Showcase

---

## 👤 Autor

**H3nri5H** (Foxyy)  
DHSH - Fortgeschrittene KI-Anwendungen & Cloud & Big Data  
Januar 2026

---

## 📦 Was ist neu (Changelog)

### v1.1 (21.01.2026)
- ✅ **setup.bat**: Windows One-Click Installer
- ✅ **setup.py**: Automatisches Setup-Script
- ✅ **Tests gefixt**: Kein pytest mehr nötig, PYTHONPATH automatisch
- ✅ **requirements.txt**: Gymnasium 0.28.1 (Ray-kompatibel)
- ✅ **README**: Idiotensichere Anleitung für Anfänger

### v1.0 (21.01.2026)
- ✅ Initial Release
- ✅ Economy Environment (Gymnasium)
- ✅ Streamlit Frontend
- ✅ FastAPI Backend
- ✅ Cloud Deployment Scripts

---

**Status**: 🟢 **Production Ready** - Alle Bugs gefixt!

🎉 **Empfohlen**: `setup.bat` (Windows) oder `python setup.py`
