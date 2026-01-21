"""Training Script für Single-Agent (Regierung)

Trainiert einen PPO-Agent als Regierung.
Firmen und Haushalte sind regelbasiert.

Zeitstruktur:
- 1 Episode = 5 Jahre = 1825 Steps
- Training über mehrere Episoden

Features:
- Strg+C sicheres Beenden (speichert Model)
- Resume from Checkpoint (Training fortsetzen)
- Clean Output (keine Warnings)
- Checkpoints alle 5 Iterationen
- Permanente Speicherung in ./checkpoints/
"""

# === WARNINGS UNTERDRÜCKEN (VOR ALLEN IMPORTS!) ===
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path
import signal
from datetime import datetime

# PYTHONPATH Auto-Fix
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from envs.economy_env import EconomyEnv


# === GLOBALE VARIABLEN FÜR SIGNAL HANDLER ===
algo_global = None
best_checkpoint_global = None


def signal_handler(sig, frame):
    """Handler für Strg+C: Speichert Model vor Exit"""
    print("\n\n" + "="*60)
    print("⚠️  Strg+C erkannt! Training wird sicher beendet...")
    print("="*60)
    
    if algo_global is not None:
        print("\n💾 Speichere finales Model...")
        try:
            final_checkpoint = algo_global.save()
            checkpoint_path = final_checkpoint.checkpoint.path
            print(f"✅ Model gespeichert: {checkpoint_path}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern: {e}")
            if best_checkpoint_global:
                print(f"💾 Letzter Checkpoint verfügbar: {best_checkpoint_global}")
        
        print("\n🛝 Ray wird heruntergefahren...")
        algo_global.stop()
        ray.shutdown()
    
    print("\n✅ Training sicher beendet!\n")
    sys.exit(0)


def find_latest_checkpoint():
    """Findet den neuesten Checkpoint im ./checkpoints Ordner
    
    Returns:
        str | None: Pfad zum Checkpoint oder None
    """
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        return None
    
    # Suche nach checkpoint_XXXXXX Ordnern
    checkpoints = []
    for ckpt_dir in checkpoint_dir.glob("checkpoint_*"):
        if ckpt_dir.is_dir():
            try:
                # Extrahiere Iteration-Nummer
                iteration = int(ckpt_dir.name.split('_')[1])
                checkpoints.append((ckpt_dir, iteration))
            except:
                continue
    
    if not checkpoints:
        return None
    
    # Neuesten Checkpoint zurückgeben
    latest = max(checkpoints, key=lambda x: x[1])
    return str(latest[0])


def train_government(num_iterations=20, max_years=5, resume_from_checkpoint=None):
    """
    Trainiere Regierungs-Agent
    
    Args:
        num_iterations: Anzahl Training-Iterationen
        max_years: Jahre pro Episode
        resume_from_checkpoint: Pfad zu Checkpoint (None = von vorne)
        
    Returns:
        str: Pfad zum finalen Checkpoint
    """
    global algo_global, best_checkpoint_global
    
    # Checkpoint-Verzeichnis erstellen
    checkpoint_base = Path("checkpoints")
    checkpoint_base.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("🧠 VWL-RL Training: Regierungs-Agent (Single-Agent)")
    print("="*60)
    print(f"Zeitstruktur: {max_years} Jahre/Episode = {365*max_years} Steps")
    print(f"Training: {num_iterations} Iterationen")
    print(f"Checkpoints: ./checkpoints/")
    
    if resume_from_checkpoint:
        print(f"\n🔄 Resume: Training wird fortgesetzt")
        print(f"   Checkpoint: {resume_from_checkpoint}")
    else:
        print(f"\n🆕 Neues Training (von Null)")
    
    print(f"\n🚫 Strg+C: Sicheres Beenden (speichert Model)")
    print("="*60 + "\n")
    
    # Strg+C Handler registrieren
    signal.signal(signal.SIGINT, signal_handler)
    
    # Ray initialisieren
    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        log_to_driver=False,
        logging_level="ERROR"
    )
    
    # PPO Config
    config = (
        PPOConfig()
        .environment(env=EconomyEnv, env_config={'max_years': max_years})
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            rollout_fragment_length="auto"
        )
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5
        )
        .resources(
            num_gpus=0
        )
        .reporting(
            min_sample_timesteps_per_iteration=4000
        )
        .debugging(
            log_level="ERROR"
        )
    )
    
    # Algo erstellen oder laden
    if resume_from_checkpoint:
        print("📥 Lade Checkpoint...")
        algo = PPO.from_checkpoint(resume_from_checkpoint)
        print("✅ Checkpoint geladen!\n")
    else:
        algo = config.build()
    
    algo_global = algo  # Für Signal Handler
    
    print("🚀 Training startet...\n")
    
    # Training Loop
    best_reward = float('-inf')
    best_checkpoint_global = None
    
    try:
        for i in range(num_iterations):
            result = algo.train()
            
            # Metriken extrahieren
            episode_reward_mean = result.get('episode_reward_mean', 0)
            episode_len_mean = result.get('episode_len_mean', 0)
            total_iterations = result.get('training_iteration', i+1)
            
            # Output
            print(f"📊 Iteration {total_iterations:3d} | "
                  f"Reward: {episode_reward_mean:9.2f} | "
                  f"Years: {episode_len_mean/365:.1f}")
            
            # Checkpoint bei Verbesserung
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                
                # Speichere in permanentes Verzeichnis
                checkpoint_path = checkpoint_base / f"checkpoint_{total_iterations:06d}"
                checkpoint_result = algo.save(checkpoint_dir=str(checkpoint_path))
                best_checkpoint_global = str(checkpoint_path)
                
                print(f"   ✅ Neuer Bestwert! Checkpoint: {checkpoint_path.name}")
            
            # Alle 5 Iterationen: Checkpoint
            if total_iterations % 5 == 0:
                checkpoint_path = checkpoint_base / f"checkpoint_{total_iterations:06d}"
                algo.save(checkpoint_dir=str(checkpoint_path))
                print(f"   💾 Checkpoint ({total_iterations})\n")
        
        # Normale Beendigung
        final_path = checkpoint_base / f"checkpoint_final"
        algo.save(checkpoint_dir=str(final_path))
        
        print("\n" + "="*60)
        print("✅ Training abgeschlossen!")
        print(f"Bester Reward: {best_reward:.2f}")
        print(f"Final Checkpoint: {final_path}")
        print("="*60 + "\n")
        
        final_checkpoint = str(final_path)
        
    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt")
        final_checkpoint = best_checkpoint_global
    
    finally:
        # Aufräumen
        algo.stop()
        ray.shutdown()
    
    return final_checkpoint


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 TRAININGS-KONFIGURATION")
    print("="*60)
    
    # === HIER KONFIGURIEREN ===
    NUM_ITERATIONS = 20      # Erhöhe für Über-Nacht-Training (z.B. 200)
    MAX_YEARS = 5            # Jahre pro Episode (z.B. 10 für länger)
    RESUME = True            # True = fortsetzen, False = neu starten
    
    print(f"Iterationen: {NUM_ITERATIONS}")
    print(f"Jahre/Episode: {MAX_YEARS}")
    print(f"Resume: {'Ja (sucht letzten Checkpoint)' if RESUME else 'Nein (neues Training)'}")
    print(f"Geschätzte Dauer: ~{NUM_ITERATIONS * 0.25:.0f} Minuten")
    print("\nTipp: Für Über-Nacht-Training:")
    print("  NUM_ITERATIONS = 200")
    print("  MAX_YEARS = 10")
    print("  RESUME = True")
    print("="*60 + "\n")
    
    # Checkpoint finden falls Resume aktiviert
    checkpoint = None
    if RESUME:
        checkpoint = find_latest_checkpoint()
        if checkpoint:
            print(f"🔍 Checkpoint gefunden: {checkpoint}")
        else:
            print("⚠️  Kein Checkpoint gefunden, starte neues Training")
    
    # Training starten
    final_checkpoint = train_government(
        num_iterations=NUM_ITERATIONS,
        max_years=MAX_YEARS,
        resume_from_checkpoint=checkpoint
    )
    
    print(f"\n🎯 Nächste Schritte:")
    print(f"1. Model evaluieren: python tests/test_scenarios.py")
    print(f"2. Frontend testen: streamlit run frontend/app.py")
    print(f"3. Weiter trainieren: RESUME = True setzen")
    print(f"4. Checkpoints: ./checkpoints/\n")
