# Training Scripts

Dieses Verzeichnis enthält alle Training-Scripte für die Multi-Agent Wirtschafts-Simulation.

---

## 🚀 Quick Start: Multi-Agent Setup testen

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Quick Test (empfohlen vor erstem Training)

```bash
python train/quick_test.py
```

**Was wird getestet:**
- ✅ Environment initialisiert korrekt
- ✅ Alle 15 Agents vorhanden (10 Haushalte + 5 Firmen)
- ✅ Actions funktionieren
- ✅ Mini-Training läuft (100 steps)

**Output (sollte so aussehen):**
```
============================================================
🧪 QUICK TEST: Multi-Agent Economy
============================================================

[1/4] Environment initialisieren...
      ✅ 15 Agents gefunden
         - Haushalte: 10
         - Firmen: 5

[2/4] Environment Reset...
      ✅ Observations: 15 agents

[3/4] Step mit Random Actions...
      ✅ Step erfolgreich

[4/4] Mini-Training starten (100 steps)...
      Training läuft...
      ✅ Training erfolgreich!

============================================================
✅ ALLE TESTS BESTANDEN!
============================================================
```

---

## 🏋️ Volles Training

### Lokales Training starten

```bash
# Kurzes Test-Training (10k steps ≈ 5 Minuten)
python train/train_local.py --timesteps 10000

# Mittleres Training (100k steps ≈ 1 Stunde)
python train/train_local.py --timesteps 100000

# Volles Training (1.25M steps = 5 Jahre ≈ 6-8 Stunden)
python train/train_local.py --timesteps 1250000
```

### Parameter

```bash
python train/train_local.py \
  --timesteps 100000 \           # Anzahl Training-Steps
  --checkpoint-freq 10 \         # Alle 10 Iterationen Checkpoint
  --output-dir ./ray_results \  # Output-Verzeichnis
  --num-workers 2 \              # Anzahl Worker
  --num-gpus 0                   # Anzahl GPUs (0 = CPU)
```

### Training-Output

Während des Trainings siehst du:

```
+-------------------------+------------+
| Trial name              | status     |
+-------------------------+------------+
| PPO_economy_xxx         | RUNNING    |
+-------------------------+------------+
| episode_reward_mean     | 5.23       |
| episodes_this_iter      | 8          |
| timesteps_total         | 2000       |
+-------------------------+------------+
```

**Wichtige Metriken:**
- `episode_reward_mean`: Durchschnittlicher Reward (sollte steigen!)
- `episodes_this_iter`: Anzahl abgeschlossene Episoden
- `timesteps_total`: Fortschritt im Training

---

## 💾 Checkpoints

Models werden automatisch gespeichert in:

```
ray_results/
└── economy_training/
    └── PPO_economy_xxx/
        ├── checkpoint_000010/  # Nach 10 Iterationen
        ├── checkpoint_000020/
        └── checkpoint_000030/
```

### Checkpoint laden und weitertrainieren

```python
from ray.rllib.algorithms.ppo import PPO

algo = PPO.from_checkpoint("ray_results/economy_training/PPO_xxx/checkpoint_000010")
result = algo.train()  # Weitertrainieren
```

---

## 🐞 Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'ray'`

```bash
pip install ray[rllib]==2.9.0 torch==2.1.0
```

### Problem: `FileNotFoundError: configs/agent_config.yaml not found`

```bash
# Script muss aus Root-Verzeichnis ausgeführt werden!
cd VWL-RL-Cloud
python train/train_local.py
```

### Problem: Training ist sehr langsam

```bash
# Mehr Worker nutzen (max = CPU-Kerne - 1)
python train/train_local.py --num-workers 4

# Oder kleinere Batch-Size (in train_local.py ändern)
```

### Problem: `OutOfMemoryError`

```bash
# Weniger Worker
python train/train_local.py --num-workers 1

# Oder kleinere Batch-Size in train_local.py:
train_batch_size=2000  # statt 4000
```

---

## 📊 TensorBoard

Training visualisieren (optional):

```bash
tensorboard --logdir=ray_results/economy_training
```

Dann im Browser: `http://localhost:6006`

---

## 📋 Nächste Schritte nach erfolgreichem Training

1. **Evaluation:** Model testen mit Evaluations-Script (kommt noch)
2. **Versionierung:** Bestes Model als `v1.0` taggen
3. **Dokumentation:** `DOCUMENTATION.md` updaten mit Ergebnissen
4. **Cloud:** Training auf GCP laufen lassen (später)

---

## 📝 Logs

Alle Logs findest du in:

```
ray_results/economy_training/PPO_xxx/
├── progress.csv          # Training-Metriken als CSV
├── result.json          # Detaillierte Results
└── events.out.tfevents  # TensorBoard-Logs
```

---

**Status:** 🟢 Multi-Agent Setup funktioniert, Ready for Training!
