# VWL-RL-Cloud

🎓 **Volkswirtschafts-Simulation mit Multi-Agent Reinforcement Learning + Cloud Deployment**  
DHSH Modul: Fortgeschrittene KI-Anwendungen & Cloud & Big Data

## 🎯 Projekt-Übersicht

**Multi-Agent RL-System:**
- 🏢 **10 Firmen-Agents** (RL): Entscheiden über Preise, Löhne, Mitarbeiteranzahl
- 🏠 **50 Haushalte** (regelbasiert): Konsum & Sparen basierend auf Einkommen
- 🏛️ **1 Regierungs-Agent** (RL): Steuerpolitik, Staatsausgaben, Zinssätze

**Cloud-Architektur:**
- ⚡ **Zustandslos**: Streamlit Frontend (User-Interface)
- 🧠 **Zustandsbehaftet**: FastAPI Backend (RL-Inference, Simulation)
- ☁️ **Google Cloud**: Cloud Run (Frontend), Cloud Run Jobs (Backend), Cloud Storage (Models)

## 🚀 Quick Start (Lokal)

### 1. Repository klonen
```bash
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud
code .  # VS Code öffnen
```

### 2. Python 3.11 venv erstellen
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

**VS Code:** `Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv\Scripts\python.exe`

### 3. Dependencies installieren
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Tests & Training
```bash
# Environment testen
python -c "import ray; from ray.rllib.algorithms.ppo import PPOConfig; print('✅ RLlib ready:', ray.__version__)"

# Training starten (kommt in nächstem Commit)
# python train/train_marl.py --steps=5000

# Frontend starten (kommt in nächstem Commit)
# streamlit run frontend/app.py
```

## 📁 Projekt-Struktur (wird erstellt)

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
├── agents/                      # Agent-Definitionen
│   ├── __init__.py
│   ├── firm_agent.py           # Firmen-RL-Agent
│   ├── household_agent.py      # Regelbasierter Haushalt
│   └── government_agent.py     # Regierungs-RL-Agent
│
├── train/                       # Training Scripts
│   ├── __init__.py
│   ├── train_marl.py           # Multi-Agent Training
│   └── train_single.py         # Single-Agent Test
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
│   ├── test_env.py
│   └── test_scenarios.py
│
└── deploy/                      # Cloud Deployment
    ├── deploy.sh               # GCP Deploy Script
    └── cloudbuild.yaml         # CI/CD Config
```

## 🎓 Anforderungen (Module)

### Fortgeschrittene KI-Anwendungen
- ✅ Multi-Agent Reinforcement Learning (RLlib)
- ✅ Custom Gymnasium Environment
- ✅ PPO-Algorithmus für Firmen & Regierung
- ✅ Reward-Shaping & Normalisierung

### Cloud & Big Data
- ✅ Zustandslose Komponente (Streamlit Frontend)
- ✅ Zustandsbehaftete Komponente (FastAPI Backend mit RL-Model)
- ✅ Google Cloud Platform Integration
- ✅ Container-Deployment (Docker)

## 📊 Nächste Schritte

Die folgenden Dateien werden in den nächsten Commits hinzugefügt:

1. ✅ **requirements.txt, .gitignore, README.md** (dieser Commit)
2. ⏳ **envs/economy_env.py** - Hauptsimulation
3. ⏳ **train/train_marl.py** - Training-Script
4. ⏳ **frontend/app.py** - Streamlit UI
5. ⏳ **backend/serve.py** - FastAPI Backend
6. ⏳ **Dockerfiles & deploy.sh** - Cloud-Deployment

## 📝 Installation für Dozenten

```bash
# Einmaliges Setup (5 Minuten)
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Simulation starten (sobald Code committed)
streamlit run frontend/app.py
```

## 🔧 Troubleshooting

| Problem | Lösung |
|---------|--------|
| `py -3.11` nicht gefunden | Python 3.11 von python.org installieren |
| Ray/RLlib Fehler | `pip install "ray[rllib]==2.10.0"` |
| Numpy/PyArrow Konflikt | Versions-Pins in requirements.txt halten |
| VS Code erkennt .venv nicht | `Ctrl+Shift+P` → "Python: Select Interpreter" |

## 📚 Dokumentation

- [Ray RLlib Docs](https://docs.ray.io/en/latest/rllib/index.html)
- [Gymnasium API](https://gymnasium.farama.org/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Google Cloud Run](https://cloud.google.com/run/docs)

## 👤 Autor

**H3nri5H** (Foxyy)  
DHSH - Fortgeschrittene KI-Anwendungen  
Januar 2026

---

**Status:** 🟡 In Entwicklung - Code folgt in nächsten Commits!
