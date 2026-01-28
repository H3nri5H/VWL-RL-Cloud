# VWL-RL-Cloud

**Multi-Agent Reinforcement Learning fuer Volkswirtschafts-Simulation**  
DHSH Module: Fortgeschrittene KI-Anwendungen & Cloud & Big Data | Januar 2026

---

## Ueberblick

Simulation einer Volkswirtschaft mit RL-Agents:
- **10 Haushalte** (konsumieren, arbeiten)
- **5 Unternehmen** (produzieren, stellen ein, setzen Preise)
- **Kein Staat** (erstmal - fokus auf Basis-Interaktion)

Jeder Agent wird von einem eigenen RL-Model gesteuert.

---

## Quick Start (Lokal)

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
[OK] Initiale Bedingungen erstellt (seed=42):
     Haushalte: 10 mit Cash 1200 EUR - 4800 EUR
     Firmen: 5 mit Kapital 120000 EUR - 480000 EUR

[TEST] Testing SimpleEconomyEnv...
[OK] Reset successful
...
```

### 4. Training (kommt spaeter)

```bash
# Lokal trainieren (wenn implementiert)
python train/train_local.py --version v1.0

# Model liegt dann in: models/v1.0.zip
```

---

## Projekt-Struktur

```
VWL-RL-Cloud/
├── configs/
│   └── agent_config.yaml         # Startbedingungen (Min/Max fuer alle)
│
├── envs/
│   └── simple_economy_env.py     # Gymnasium Environment (Haushalte+Firmen)
│
├── tests/
│   └── test_simple_env.py        # Test fuer fixe Startbedingungen + Seeds
│
├── train/                       # TODO: Training Scripts
│   ├── train_local.py
│   └── train_cloud.py
│
├── backend/                     # TODO: FastAPI (zustandsbehaftet)
│   ├── serve.py
│   └── Dockerfile
│
├── frontend/                    # TODO: Streamlit (zustandslos)
│   ├── app.py
│   └── Dockerfile
│
├── deploy/                      # TODO: Cloud Deployment
│   ├── terraform/
│   └── k8s/
│
├── models/                      # Models werden hier gespeichert
│   └── latest_model.zip         # Fuer Dozenten (kommt spaeter)
│
├── DOCUMENTATION.md            # Was wurde gemacht + Warum
└── README.md                   # Diese Datei (Setup-Anleitung)
```

---

## Konfiguration

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
    min: 100000  # 100k EUR
    max: 500000  # 500k EUR
  
  initial_employees:
    min: 3
    max: 8

simulation:
  days_per_year: 250  # Betriebstage
  max_years: 5        # Training-Dauer
```

**Wichtig:** Diese Werte werden **einmal beim Init** gezogen und bleiben dann **ueber alle Episoden fix**!

---

## Wie funktioniert das?

### Startbedingungen mit Seeds

#### Reproduzierbare Experimente (mit Seed)

```python
# Mit festem Seed - IMMER gleiche Startbedingungen
env = SimpleEconomyEnv(seed=42)

# Experiment 1
env = SimpleEconomyEnv(seed=42)
env.reset()
# Haushalt_0: 3244.56 EUR

# Experiment 2 (Tage spaeter)
env = SimpleEconomyEnv(seed=42)
env.reset()
# Haushalt_0: 3244.56 EUR  <-- GLEICH!
```

**Nutzen:**
- Experimente sind reproduzierbar
- Papers koennen repliziert werden
- Debugging einfacher

#### Zufaellige Variation (ohne Seed)

```python
# Ohne Seed - Jedes Mal anders
env1 = SimpleEconomyEnv()
# Haushalt_0: 2500 EUR

env2 = SimpleEconomyEnv()
# Haushalt_0: 4123 EUR  <-- ANDERS!
```

**Nutzen:**
- Generalisierung testen
- Robustheit pruefen

### Episoden

```python
# Episode 1
obs = env.reset()  # Haushalte/Firmen bei Startwerten
for day in range(250):  # 1 Jahr
    action = agent.predict(obs)
    obs, reward, done, info = env.step(action)

# Episode 2
obs = env.reset()  # WIEDER bei Startwerten (NICHT weiterfuehren!)
# Haushalt_0 startet wieder mit 2500 EUR (oder was der Seed vorgab)
```

**Wichtig:** 
- Gewinn aus Episode 1 wird **NICHT** in Episode 2 uebernommen
- Jede Episode startet "frisch" mit den fixen Startwerten
- Aber: RL-Agent **lernt** aus allen Episoden!

---

## Development

### Tests ausfuehren

```bash
python tests/test_simple_env.py
```

**Prueft:**
- Startbedingungen bleiben ueber Episoden fix
- Seeds funktionieren (Reproduzierbarkeit)
- Environment kann resetten
- Steps funktionieren

### Environment direkt nutzen

```python
from envs.simple_economy_env import SimpleEconomyEnv

# Mit Seed (reproduzierbar)
env = SimpleEconomyEnv(seed=42)

# Ohne Seed (random)
env = SimpleEconomyEnv()

obs, info = env.reset()

# Manuelle Aktionen
for _ in range(10):
    action = env.action_space.sample()  # Zufaellige Action
    obs, reward, done, info = env.step(action)
    print(f"Day {info['day']}: Reward={reward}")
```

---

## Dokumentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Was wurde gemacht + Design-Entscheidungen
- **[configs/agent_config.yaml](configs/agent_config.yaml)** - Parameter-Dokumentation

---

## Team

**H3nri5H** (Foxyy)  
DHSH - Januar 2026

---

## Status

**Version:** 0.1 - Basis-Setup  
**Stand:** 28.01.2026

**Implementiert:**
- Config mit Min/Max-Bereichen
- Simple Environment (Haushalte + Firmen)
- Fixe Startbedingungen mit Seed-Support
- Tests (inkl. Seed-Reproduzierbarkeit)

**Naechste Schritte:**
1. Wirtschafts-Logik implementieren (Produktion, Konsum, Markt)
2. Action/Observation Spaces definieren
3. Reward-Funktionen designen
4. Lokales Training testen
5. Backend/Frontend implementieren
6. Cloud Deployment

---

**Fuer Dozenten:** Ein trainiertes Model wird spaeter in `models/latest_model.zip` hochgeladen, sodass kein Training notwendig ist.
