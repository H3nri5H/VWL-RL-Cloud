# 📝 VWL-RL-Cloud: Entwicklungs-Dokumentation

**Projekt:** Volkswirtschafts-Simulation mit Multi-Agent RL  
**Modul:** Fortgeschrittene KI-Anwendungen & Cloud & Big Data  
**Zeitraum:** Januar 2026  
**Autor:** H3nri5H (Foxyy)

---

## 🎯 Projekt-Ziel

**Vision:**  
Multi-Agent Reinforcement Learning System zur Simulation makroökonomischer Prozesse. Drei Agent-Typen (Firmen, Haushalte, Staat) lernen eigenständig optimale Strategien in einer simulierten Volkswirtschaft.

**Module-Anforderungen:**
- ✅ **KI-Anwendungen**: Multi-Agent RL mit Ray RLlib
- ✅ **Cloud**: Zustandslose (Frontend) + Zustandsbehaftete (Backend) Komponenten auf GCP

---

## 📅 Entwicklungs-Timeline

### **Phase 1: Setup & Grundstruktur** (21.01.2026)

#### ✅ **Morgens: Repository & Environment**
- Repository erstellt: `github.com/H3nri5H/VWL-RL-Cloud`
- Basis-Environment implementiert (`envs/economy_env.py`)
- Gymnasium-kompatible Struktur
- Firmen + Haushalte als Dictionaries
- Regierung als RL-Agent (Single-Agent)

**Technische Entscheidungen:**
- Python 3.11 (beste Ray-Kompatibilität)
- Ray RLlib 2.10 (stabiler als 2.x+)
- Gymnasium 0.28.1 (Ray-Requirement)
- PPO-Algorithmus (stabil für continuous actions)

---

#### ✅ **Mittags: Setup-Automatisierung**

**Problem:** Dependency-Konflikte bei Installation
- Ray 2.10 braucht Gymnasium 0.28.1 (nicht 0.29.1)
- PYTHONPATH-Probleme bei Tests
- pytest-Dependency unnötig

**Lösung:**
- `setup.py` erstellt (automatisches Setup)
- `setup.bat` erstellt (Windows One-Click)
- `requirements.txt` gefixt (korrekte Gymnasium-Version)
- Tests ohne pytest neu geschrieben (PYTHONPATH auto-fix)
- README komplett überarbeitet (idiotensicher)

**Ergebnis:** Setup in 10 Minuten für komplette Anfänger möglich

---

#### ✅ **Nachmittags: Streamlit Frontend**
- Interaktive Web-UI implementiert
- 3 Parameter-Sliders (Steuern, Ausgaben, Zinsen)
- 4 Szenarien (Normal, Rezession, Boom, Inflation)
- Live-Plots mit Plotly (BIP, Inflation, Arbeitslosigkeit)
- 100-Step Simulation pro Klick

**Architektur-Typ:** Zustandslos (Environment wird bei jedem Request neu erstellt)

---

#### ✅ **Abends: Zeitstruktur & Dokumentation** (18:00 - 18:10 Uhr)

**Diskussion: Multi-Agent Architektur**

**Problem identifiziert:**  
"Moving Target Problem" - Agents beeinflussen sich gegenseitig, Environment wird non-stationary

**Lösungsansätze diskutiert:**
1. CTDE (Centralized Training, Decentralized Execution) - MAPPO
2. Self-Play (wie AlphaGo)
3. Population-Based Training
4. Curriculum Learning (stufenweises Trainieren)

**Entscheidung:**  
- START: Single-Agent (nur Staat) mit regelbasierten Firmen/Haushalten
- SPÄTER: Multi-Agent wenn Zeit bleibt
- GRUND: Stabiler, weniger Risiko, für Präsentation ausreichend

**Implementiert:**
- ✅ Zeitstruktur: 1 Step = 1 Tag, 1 Episode = 365 Tage = 1 Jahr
- ✅ `current_day` & `current_year` Tracking
- ✅ Jahresabschluss-Logik (Metriken sammeln)
- ✅ Episode endet nach 5 Jahren (trainierbar auf 10-50 Jahre)
- ✅ DEVELOPMENT.md für Dokumentation angelegt

---

## 🧠 Architektur-Entscheidungen

### **Environment Design**

**Aktuell (Phase 1 - Single Agent):**
```
Regierung (RL-Agent)
  ↓ Actions: [tax_rate, gov_spending, interest_rate]
  ↓
Wirtschaft (Simulation)
  ├─ 10 Firmen (regelbasiert)
  ├─ 50 Haushalte (regelbasiert) 
  └─ Markt (Clearing)
  ↓
Observations: [bip, inflation, unemployment, debt, interest]
Reward: f(bip_growth, unemployment, inflation, deficit)
```

**Geplant (Phase 2 - Multi Agent):**
```
Firmen (10x RL-Agents)
  Actions: [price, wage, hire/fire, investment]
  Reward: profit

Haushalte (50x RL-Agents)
  Actions: [consumption_rate, savings_rate, job_search]
  Reward: consumption + savings - unemployment_penalty

Regierung (1x RL-Agent)
  Actions: [tax, spending, interest, unemployment_aid]
  Reward: gdp_growth - unemployment - abs(inflation-2%)
```

---

### **Reward-Funktion (Staat)**

**Aktuell:**
```python
reward = (
    + bip_growth * 10        # BIP-Wachstum belohnen
    - unemployment * 20       # Arbeitslosigkeit stark bestrafen
    - abs(inflation) * 15     # Inflation (egal ob +/-) bestrafen
    - abs(deficit) * 0.01     # Defizit leicht bestrafen
)
```

**Rationale:**
- BIP-Wachstum = primäres Ziel (Wohlstand)
- Arbeitslosigkeit = soziales Problem (hohe Strafe)
- Inflation = Stabilität (symmetrisch bestraft)
- Defizit = nachhaltig (kleine Strafe, nicht primär)

**Balancing:** Wird während Training angepasst falls nötig

---

### **Zeitstruktur**

**Mapping:**
- 1 Step = 1 Tag
- 365 Steps = 1 Jahr = 1 Episode
- Training über mehrere Episoden = mehrere Jahre

**Vorteile:**
- Realistische Zeitskalen
- Saisonale Effekte möglich (später)
- Jahresabschlüsse für Metriken
- Vergleichbar mit realen Daten

**Episode-Terminierung:**
- Nach 5 Jahren (1825 Steps) - Trainingsdefault
- Konfigurierbar für längere Simulationen

---

## 🐛 Bekannte Probleme & Lösungen

### **Problem 1: Gymnasium Version Conflict**
**Symptom:** `ResolutionImpossible` bei pip install  
**Ursache:** Ray 2.10 braucht exakt Gymnasium 0.28.1  
**Lösung:** requirements.txt auf 0.28.1 gefixt  
**Status:** ✅ Gelöst

### **Problem 2: ModuleNotFoundError 'envs'**
**Symptom:** Import Error bei Tests  
**Ursache:** PYTHONPATH nicht gesetzt  
**Lösung:** Auto-fix in test_env.py (`sys.path.insert`)  
**Status:** ✅ Gelöst

### **Problem 3: Non-Stationary Environment (Multi-Agent)**
**Symptom:** Agents' Strategien werden ständig invalidiert  
**Ursache:** Gegenseitige Beeinflussung der Agents  
**Lösung:** Start mit Single-Agent, später MAPPO/Self-Play  
**Status:** 🟡 Design-Entscheidung getroffen

---

## 📝 TODO / Nächste Schritte

### **Sofort (Diese Woche)**
- [ ] Training mit neuer Zeitstruktur testen
- [ ] Hyperparameter tunen (Learning Rate, Batch Size)
- [ ] Längeres Training (10+ Jahre)
- [ ] Reward-Funktion evaluieren & ggf. anpassen

### **Nächste Woche**
- [ ] FastAPI Backend implementieren (Model Loading)
- [ ] TensorBoard Logging aktivieren
- [ ] Docker Images bauen
- [ ] GCP Account einrichten

---

**Letztes Update:** 21.01.2026, 18:10 Uhr  
**Status:** 🟢 Active Development  
**Nächster Meilenstein:** Training mit Zeitstruktur
