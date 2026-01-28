# VWL Multi-Agent RL - Entwicklungsdokumentation

Dieses Dokument beschreibt **WAS** gemacht wurde und **WARUM** bestimmte Entscheidungen getroffen wurden.

---

## Architektur-Entscheidungen

### Warum kein Staat am Anfang?

**Entscheidung:** Das Projekt startet NUR mit Haushalten und Unternehmen.

**Begründung:**
- Fokus auf Basis-Interaktion: Haushalte ↔ Unternehmen
- Einfacheres Environment für erste Tests
- Wirtschaftskreislauf ist bereits mit 2 Akteuren komplett
- Staat kann später als 3. Agent hinzugefügt werden

**Status:** ✅ Implementiert in `envs/rllib_economy_env.py`

---

### Startbedingungen: Zufällig pro Episode

**Entscheidung:** Startbedingungen werden **bei jedem Episode-Reset neu gezogen**.

**Was heißt das konkret?**
```python
# Bei jedem env.reset():
Episode 1: Haushalt_0 startet mit 2500€ (zufällig aus [1000-5000€])
Episode 2: Haushalt_0 startet mit 4100€ (neu gezogen!)
Episode 3: Haushalt_0 startet mit 1800€ (neu gezogen!)
```

**Begründung:**
1. **Robustere Policy:** Agent lernt mit verschiedenen Startbedingungen umzugehen
2. **Generalisierung:** Policy funktioniert für arme UND reiche Haushalte
3. **Realitätsnäher:** In der echten Welt variieren Startbedingungen auch
4. **RL-Best-Practice:** Variation fördert besseres Lernen

**Reproduzierbarkeit:** Durch Seed-Parameter bei `reset()` steuerbar

**Status:** ✅ Implementiert in `envs/rllib_economy_env.py`

---

### Multi-Agent Setup

**Entscheidung:** Jeder Haushalt und jede Firma ist ein eigenständiger RL-Agent.

**Architektur:**
- **10 Haushalte** = 10 individuelle Agents (`household_0` bis `household_9`)
- **5 Firmen** = 5 individuelle Agents (`firm_0` bis `firm_4`)
- **Gesamt:** 15 Agents trainieren parallel

**Policy Sharing:**
- Alle Haushalte teilen sich **eine Policy** (`household_policy`)
- Alle Firmen teilen sich **eine Policy** (`firm_policy`)
- → Nur 2 Policies für 15 Agents (effizient!)

**Begründung:**
- Parameter-Effizienz: Haushalte ähneln sich, müssen nicht individuell trainiert werden
- Schnelleres Lernen: Mehr Erfahrungen pro Policy-Update
- Skalierbar: Später 100 Haushalte ohne neue Policy

**Status:** ✅ Implementiert mit RLlib MultiAgentEnv

---

### Konfiguration via YAML

**Entscheidung:** Alle Agent-Parameter in separater Config-Datei.

**Struktur:**
```yaml
households:
  count: 10
  initial_cash:
    min: 1000    # Alle Haushalte ziehen aus diesem Bereich
    max: 5000

firms:
  count: 5
  initial_capital:
    min: 100000  # Alle Firmen ziehen aus diesem Bereich
    max: 500000
```

**Begründung:**
- Einfache Anpassung ohne Code-Änderungen
- Dokumentation der Parameter
- Verschiedene Configs für Experimente möglich
- Standard in ML-Projekten

**Status:** ✅ Implementiert in `configs/agent_config.yaml`

---

### Wirtschaftslogik: Erst simpel, dann realistisch

**Entscheidung Phase 1:** Minimale Wirtschaftslogik für erste Tests.

**Aktuell implementiert:**
- Haushalte: Wählen Konsumquote (0-100% vom Cash)
- Firmen: Produzieren, setzen Preise, stellen ein/entlassen
- Bankrott-Mechanismus: Cash < 0 → Agent bankrott

**Noch NICHT implementiert (kommt später):**
- ❌ Arbeitsmarkt (Haushalte erhalten Lohn)
- ❌ Gütermarkt (Produktion → Verkauf → Konsum)
- ❌ Preismechanismus (Angebot/Nachfrage)
- ❌ Geldfluss-Kreislauf (Lohn → Konsum → Revenue)

**Begründung:**
1. **Erst Multi-Agent zum Laufen bringen** → Dann verfeinern
2. Simple Version ist leichter zu debuggen
3. Iterative Entwicklung: Komplexe Features schrittweise hinzufügen

**Nächster Schritt:** Nach erfolgreichem ersten Training realistischere Wirtschaftslogik einbauen

**Status:** 🟡 Phase 1 (simpel) implementiert, Phase 2 (realistisch) geplant

---

## Was wurde implementiert

### Version 0.2 - Multi-Agent RLlib Integration (28.01.2026)

**Erstellt:**
1. `envs/rllib_economy_env.py` - RLlib-kompatibler Multi-Agent Wrapper
2. `train/train_local.py` - PPO Training-Script
3. `train/quick_test.py` - Test-Script für Multi-Agent Setup
4. `train/README.md` - Training-Dokumentation

**Features:**
- ✅ 15 individuelle Agents (10 Haushalte + 5 Firmen)
- ✅ Policy Sharing (2 Policies für 15 Agents)
- ✅ Separate Action/Observation Spaces
- ✅ PPO-Algorithmus konfiguriert
- ✅ Checkpoint-System
- ✅ Command-Line Interface

**Testing:**
```bash
# Quick Test (funktioniert!)
python train/quick_test.py

# Volles Training
python train/train_local.py --timesteps 10000
```

**Status:** 🟢 **Ready for Training!**

---

### Version 0.1 - Basis-Setup (28.01.2026)

**Erstellt:**
1. `configs/agent_config.yaml` - Konfiguration mit Min/Max-Bereichen
2. `envs/simple_economy_env.py` - Simples Environment (Gymnasium-kompatibel)
3. `tests/test_simple_env.py` - Basis-Tests

**Features:**
- ✅ 10 Haushalte mit Cash aus [1000-5000€]
- ✅ 5 Firmen mit Kapital aus [100k-500k€]
- ✅ Bankrott-Mechanismus
- ✅ 250 Tage pro Jahr, 5 Jahre Training

---

## Technische Details

### Zeitstruktur

- **1 Step** = 1 Betriebstag
- **1 Episode** = 250 Tage = 1 Wirtschaftsjahr
- **Training** = 5 Jahre = 1250 Episoden = 312.500 Steps

### Action Spaces

**Haushalte:**
```python
Box(low=[0.0], high=[1.0])  # Konsumquote (0-100%)
```

**Firmen:**
```python
Box(
    low=[0.0, 5.0, -2.0],   # [Produktion, Preis, Mitarbeiteränderung]
    high=[200.0, 15.0, 2.0]
)
```

### Observation Spaces

**Haushalte:**
```python
Box(
    low=[0.0, 0.0, 0.0],     # [Cash, Durchschnittspreis, Beschäftigt]
    high=[100000.0, 50.0, 1.0]
)
```

**Firmen:**
```python
Box(
    low=[0.0, 0.0, 0.0, 0.0],      # [Kapital, Lager, Mitarbeiter, Nachfrage]
    high=[1000000.0, 1000.0, 50.0, 1000.0]
)
```

### Reward-Funktionen (simpel)

**Haushalte:**
```python
if bankrupt:
    reward = -10.0
else:
    reward = consumption * 0.1 + 1.0  # Konsum + Überleben
```

**Firmen:**
```python
if bankrupt:
    reward = -10.0
else:
    reward = capital / 100000.0  # Kapital normalisiert
```

---

## Nächste Schritte

### Phase 1: Erstes Training (🔴 **JETZT**)

1. ✅ Quick Test durchführen
2. 🔴 Kurzes Training (10k steps) starten
3. 🔴 Metriken analysieren: Lernen die Agents?
4. 🔴 DOCUMENTATION.md updaten mit Ergebnissen

### Phase 2: Wirtschaftslogik verbessern

1. Arbeitsmarkt implementieren (Firmen → Löhne → Haushalte)
2. Gütermarkt implementieren (Produktion → Verkauf → Konsum)
3. Geldfluss-Kreislauf schließen
4. Reward-Funktionen anpassen

### Phase 3: Cloud Deployment

1. Backend (FastAPI) für Inference
2. Frontend (Streamlit) für Visualisierung
3. Google Cloud Platform Setup
4. CI/CD Pipeline

---

## Lessons Learned

### Was funktioniert gut
- ✅ RLlib Multi-Agent API ist sehr elegant
- ✅ Policy Sharing spart massiv Parameter
- ✅ YAML-Config macht Experimente einfach
- ✅ Quick-Test-Script verhindert lange Debug-Sessions

### Was noch offen ist
- ❓ Lernen die Agents sinnvoll mit simpler Wirtschaftslogik?
- ❓ Wie schnell konvergiert das Training?
- ❓ Brauchen wir komplexere Rewards?

---

## Git-History

### Commits (neueste zuerst)

- `09d5ecc` - docs: Add training README with quick test instructions
- `ad68eb9` - feat: Add quick test script for multi-agent setup
- `fb9079a` - feat: Add RLlib multi-agent training setup
- `617edc9` - feat: Add simple multi-agent economy environment
- `38f78bd` - feat: Add simple agent configuration

### Branches
- `main` - Hauptentwicklung

### Tags
(Kommen nach ersten erfolgreichen Trainings)

---

## Team-Notizen

**28.01.2026 - 12:57 Uhr:**
- Multi-Agent Setup vollständig implementiert
- Quick-Test-Script hinzugefügt
- Bereit für erstes Training!
- Nächster Schritt: Training durchführen und Ergebnisse dokumentieren

---

**Letztes Update:** 28.01.2026, 12:58 Uhr
