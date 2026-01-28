# VWL-RL-Cloud 🏭

**Multi-Agent Reinforcement Learning für Volkswirtschafts-Simulation**  
DHSH Module: Fortgeschrittene KI-Anwendungen & Cloud & Big Data | Januar 2026

---

## 💁 Überblick

Simulation einer Volkswirtschaft mit RL-Agents:
- **10 Haushalte** (konsumieren, arbeiten)
- **5 Unternehmen** (produzieren, stellen ein, setzen Preise)
- **Kein Staat** (erstmal - fokus auf Basis-Interaktion)

Jeder Agent wird von einem eigenen RL-Model gesteuert.

---

## 🚀 Quick Start (Lokal)

### 1. Repository klonen

```bash
git clone https://github.com/H3nri5H/VWL-RL-Cloud.git
cd VWL-RL-Cloud
```

### 2. Dependencies installieren

```bash
# Python 3.11+ required
pip install -r requirements.txt
```

### 3. Environment testen

```bash
# Basis-Test
python tests/test_simple_env.py

# Oder direkt Environment starten
python envs/simple_economy_env.py
```

**Ausgabe sollte sein:**
```
✅ Initiale Bedingungen erstellt (fix für alle Episoden):
   Haushalte: 10 mit Cash 1200€ - 4800€
   Firmen: 5 mit Kapital 120000€ - 480000€

🧪 Testing SimpleEconomyEnv...
✅ Reset successful
...
```

### 4. Training (kommt später)

```bash
# Lokal trainieren (wenn implementiert)
python train/train_local.py --version v1.0

# Model liegt dann in: models/v1.0.zip
```

---

## 📋 Projekt-Struktur

```
VWL-RL-Cloud/
├── configs/
│   └── agent_config.yaml         # ✅ Startbedingungen (Min/Max für alle)
│
├── envs/
│   └── simple_economy_env.py     # ✅ Gymnasium Environment (Haushalte+Firmen)
│
├── tests/
│   └── test_simple_env.py        # ✅ Test für fixe Startbedingungen
│
├── train/                       # ❌ TODO: Training Scripts
│   ├── train_local.py
│   └── train_cloud.py
│
├── backend/                     # ❌ TODO: FastAPI (zustandsbehaftet)
│   ├── serve.py
│   └── Dockerfile
│
├── frontend/                    # ❌ TODO: Streamlit (zustandslos)
│   ├── app.py
│   └── Dockerfile
│
├── deploy/                      # ❌ TODO: Cloud Deployment
│   ├── terraform/
│   └── k8s/
│
├── models/                      # Models werden hier gespeichert
│   └── latest_model.zip         # Für Dozenten (kommt später)
│
├── DOCUMENTATION.md            # ✅ Was wurde gemacht + Warum
└── README.md                   # Diese Datei (Setup-Anleitung)
```

---

## ⚙️ Konfiguration

### Agent-Parameter anpassen

**Datei:** `configs/agent_config.yaml`

```yaml
households:
  count: 10  # Anzahl Haushalte
  
  initial_cash:
    min: 1000    # Minimum Startkapital
    max: 5000    # Maximum Startkapital

firms:
  count: 5  # Anzahl Unternehmen
  
  initial_capital:
    min: 100000  # 100k€
    max: 500000  # 500k€
  
  initial_employees:
    min: 3
    max: 8

simulation:
  days_per_year: 250  # Betriebstage
  max_years: 5        # Training-Dauer
```

**Wichtig:** Diese Werte werden **einmal beim Init** gezogen und bleiben dann **über alle Episoden fix**!

---

## 🧠 Wie funktioniert das?

### Startbedingungen

```python
# Beim Training-Start (env.__init__):
env = SimpleEconomyEnv()

# Zieht für jeden Agent zufällige Werte:
Haushalt_0: 2500€  (aus [1000-5000€])
Haushalt_1: 4200€  (aus [1000-5000€])
Firma_0: 250.000€  (aus [100k-500k€])

# Diese Werte bleiben FIX!
```

### Episoden

```python
# Episode 1
obs = env.reset()  # Haushalte/Firmen bei Startwerten
for day in range(250):  # 1 Jahr
    action = agent.predict(obs)
    obs, reward, done, info = env.step(action)

# Episode 2
obs = env.reset()  # WIEDER bei Startwerten (NICHT weiterführen!)
# Haushalt_0 startet wieder mit 2500€
```

**Wichtig:** 
- Gewinn aus Episode 1 wird **NICHT** in Episode 2 übernommen
- Jede Episode startet "frisch" mit den fixen Startwerten
- Aber: RL-Agent **lernt** aus allen Episoden!

---

## 📚 Module-Anforderungen

### Fortgeschrittene KI-Anwendungen
- ✅ Multi-Agent Reinforcement Learning
- ✅ Custom Gymnasium Environment
- ❌ RL-Training (TODO)
- ❌ Reward-Design (TODO)

### Cloud & Big Data
- ❌ Zustandslose Komponente (Frontend)
- ❌ Zustandsbehaftete Komponente (Backend mit Models)
- ❌ Cloud Deployment (GCP)
- ❌ CI/CD Pipeline

---

## 🛠️ Development

### Tests ausführen

```bash
python tests/test_simple_env.py
```

**Prüft:**
- ✅ Startbedingungen bleiben über Episoden fix
- ✅ Environment kann resetten
- ✅ Steps funktionieren

### Environment direkt nutzen

```python
from envs.simple_economy_env import SimpleEconomyEnv

env = SimpleEconomyEnv()
obs, info = env.reset()

# Manuelle Aktionen
for _ in range(10):
    action = env.action_space.sample()  # Zufällige Action
    obs, reward, done, info = env.step(action)
    print(f"Day {info['day']}: Reward={reward}")
```

---

## 📝 Dokumentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Was wurde gemacht + Design-Entscheidungen
- **[configs/agent_config.yaml](configs/agent_config.yaml)** - Parameter-Dokumentation

---

## 👥 Team

**H3nri5H** (Foxyy)  
DHSH - Januar 2026

---

## 📌 Status

**Version:** 0.1 - Basis-Setup  
**Stand:** 28.01.2026

**Implementiert:**
- ✅ Config mit Min/Max-Bereichen
- ✅ Simple Environment (Haushalte + Firmen)
- ✅ Fixe Startbedingungen
- ✅ Tests

**Nächste Schritte:**
1. Wirtschafts-Logik implementieren (Produktion, Konsum, Markt)
2. Action/Observation Spaces definieren
3. Reward-Funktionen designen
4. Lokales Training testen
5. Backend/Frontend implementieren
6. Cloud Deployment

---

**Für Dozenten:** Ein trainiertes Model wird später in `models/latest_model.zip` hochgeladen, sodass kein Training notwendig ist.
