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

**Status:** ✅ Implementiert in `envs/simple_economy_env.py`

---

### Fixe vs. Zufällige Startbedingungen

**Entscheidung:** Startbedingungen werden **einmal beim Init** zufällig gezogen und bleiben dann **über alle Episoden fix**.

**Was heißt das konkret?**
```python
# Beim Training-Start (env.__init__):
Haushalt_0: Zieht 2500€ aus [1000-5000€]
Haushalt_1: Zieht 4200€ aus [1000-5000€]
Firma_0: Zieht 250k€ aus [100k-500k€]

# Dann in JEDER Episode:
Episode 1: Haushalt_0 startet mit 2500€
Episode 2: Haushalt_0 startet mit 2500€  # IMMER GLEICH!
Episode 3: Haushalt_0 startet mit 2500€
```

**Begründung:**
1. **Wissenschaftlich sauberer:** Vergleichbarkeit zwischen Episoden
2. **Reproduzierbar:** Training kann mit Seed wiederholt werden
3. **Rollen-Analyse möglich:** "Wie verhält sich armer vs. reicher Haushalt?"
4. **Einfachere Evaluation:** Klare Zuordnung Agent → Performance

**Alternative (nicht gewählt):**
Jede Episode neue Zufallswerte → Robustere Policy, aber schlechter analysierbar

**Status:** ✅ Implementiert in `configs/agent_config.yaml` + `SimpleEconomyEnv._initialize_initial_conditions()`

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

## Was wurde implementiert

### Version 0.1 - Basis-Setup (28.01.2026)

**Erstellt:**
1. `configs/agent_config.yaml` - Konfiguration mit Min/Max-Bereichen
2. `envs/simple_economy_env.py` - Simples Environment (nur Haushalte + Firmen)
3. `tests/test_simple_env.py` - Test für fixe Startbedingungen

**Features:**
- ✅ 10 Haushalte mit Cash aus [1000-5000€]
- ✅ 5 Firmen mit Kapital aus [100k-500k€]
- ✅ Fixe Startbedingungen über Episoden
- ✅ Pleite-Mechanismus für Haushalte/Firmen
- ✅ 250 Tage pro Jahr, 5 Jahre Training

**Nicht implementiert (kommt später):**
- ❌ Wirtschafts-Logik (Produktion, Konsum, Markt)
- ❌ RL-Algorithmus Integration
- ❌ Reward-Funktion
- ❌ Multi-Agent Policies
- ❌ Cloud Deployment

**Nächste Schritte:**
1. Wirtschafts-Logik implementieren
2. Action/Observation Spaces definieren
3. Reward-Funktionen designen
4. Erstes lokales Training testen

---

## Technische Details

### Zeitstruktur

- **1 Step** = 1 Betriebstag
- **1 Episode** = 250 Tage = 1 Wirtschaftsjahr
- **Training** = 5 Jahre = 1250 Episoden = 312.500 Steps

### Environment-Lifecycle

```python
# Training-Start
env = SimpleEconomyEnv()  # Zieht initiale Bedingungen

# Episode 1
obs = env.reset()  # Setzt Agents zurück zu initialen Bedingungen
for day in range(250):
    action = agent.predict(obs)
    obs, reward, done, info = env.step(action)

# Episode 2
obs = env.reset()  # GLEICHE initiale Bedingungen wie Episode 1!
# ...
```

---

## Lessons Learned

### Was funktioniert gut
- YAML-Config ist sehr übersichtlich
- Fixe Startbedingungen machen Debugging einfacher
- Simple Basis → später erweitern ist guter Ansatz

### Was noch zu tun ist
- Wirtschafts-Logik muss noch durchdacht werden
- Multi-Agent RL-Framework entscheiden (PettingZoo vs. custom)
- Reward-Design ist kritisch für Lernerfolg

---

## Git-History

### Commits
- `38f78bd` - feat: Add simple agent configuration with min/max ranges
- `617edc9` - feat: Add simple multi-agent economy environment (no government)
- `f75d661` - feat: Add test script for simple economy environment

### Branches
- `main` - Hauptentwicklung

### Tags
(Noch keine - kommen nach ersten Trainings)

---

## Team-Notizen

*Hier können später Notizen, Ideen, TODOs eingetragen werden*

---

**Letztes Update:** 28.01.2026
