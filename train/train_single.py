"""Training Script für Single-Agent (Regierung)

Trainiert einen PPO-Agent als Regierung.
Firmen und Haushalte sind regelbasiert.

Zeitstruktur:
- 1 Episode = 5 Jahre = 1825 Steps
- Training über mehrere Episoden

Features:
- Strg+C sicheres Beenden (speichert Model vor Exit)
- Clean Output (keine Warnings)
- Checkpoints alle 5 Iterationen
"""

# === WARNINGS UNTERDRÜCKEN (VOR ALLEN IMPORTS!) ===
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path
import signal

# PYTHONPATH Auto-Fix
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import ray
from ray.rllib.algorithms.ppo import PPOConfig
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
            print(f"✅ Model gespeichert: {final_checkpoint}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern: {e}")
            if best_checkpoint_global:
                print(f"💾 Letzter Checkpoint verfügbar: {best_checkpoint_global}")
        
        print("\n🛝 Ray wird heruntergefahren...")
        algo_global.stop()
        ray.shutdown()
    
    print("\n✅ Training sicher beendet!\n")
    sys.exit(0)


def train_government(num_iterations=20, max_years=5):
    """
    Trainiere Regierungs-Agent
    
    Args:
        num_iterations: Anzahl Training-Iterationen
        max_years: Jahre pro Episode
        
    Returns:
        str: Pfad zum finalen Checkpoint
    """
    global algo_global, best_checkpoint_global
    
    print("\n" + "="*60)
    print("🧠 VWL-RL Training: Regierungs-Agent (Single-Agent)")
    print("="*60)
    print(f"Zeitstruktur: {max_years} Jahre/Episode = {365*max_years} Steps")
    print(f"Training: {num_iterations} Iterationen")
    print(f"\n🚫 Strg+C: Sicheres Beenden (speichert Model)")
    print("="*60 + "\n")
    
    # Strg+C Handler registrieren
    signal.signal(signal.SIGINT, signal_handler)
    
    # Ray initialisieren
    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        log_to_driver=False,
        logging_level="ERROR"  # Nur Errors
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
    
    # Algo erstellen
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
            
            # Output
            print(f"📊 Iteration {i+1:3d}/{num_iterations} | "
                  f"Reward: {episode_reward_mean:9.2f} | "
                  f"Years: {episode_len_mean/365:.1f}")
            
            # Checkpoint bei Verbesserung
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                best_checkpoint_global = algo.save()
                print(f"   ✅ Neuer Bestwert!")
            
            # Alle 5 Iterationen: Checkpoint
            if (i + 1) % 5 == 0:
                checkpoint = algo.save()
                print(f"   💾 Checkpoint ({i+1}/{num_iterations})\n")
        
        # Normale Beendigung
        final_checkpoint = algo.save()
        
        print("\n" + "="*60)
        print("✅ Training abgeschlossen!")
        print(f"Bester Reward: {best_reward:.2f}")
        print(f"Model gespeichert: {final_checkpoint}")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        # Sollte durch signal_handler gefangen werden
        print("\n⚠️  KeyboardInterrupt (sollte nicht hier landen)")
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
    
    print(f"Iterationen: {NUM_ITERATIONS}")
    print(f"Jahre/Episode: {MAX_YEARS}")
    print(f"Geschätzte Dauer: ~{NUM_ITERATIONS * 0.25:.0f} Minuten")
    print("\nTipp: Für Über-Nacht-Training:")
    print("  NUM_ITERATIONS = 200")
    print("  MAX_YEARS = 10")
    print("="*60 + "\n")
    
    # Training starten
    checkpoint = train_government(
        num_iterations=NUM_ITERATIONS,
        max_years=MAX_YEARS
    )
    
    print(f"\n🎯 Nächste Schritte:")
    print(f"1. Model evaluieren: python tests/test_scenarios.py")
    print(f"2. Frontend testen: streamlit run frontend/app.py")
    print(f"3. Checkpoint-Pfad: {checkpoint}\n")
