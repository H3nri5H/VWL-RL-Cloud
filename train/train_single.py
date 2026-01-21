"""Training Script für Single-Agent (Regierung)

Trainiert einen PPO-Agent als Regierung.
Firmen und Haushalte sind regelbasiert.

Zeitstruktur:
- 1 Episode = 5 Jahre = 1825 Steps
- Training über mehrere Episoden
"""

import sys
from pathlib import Path

# PYTHONPATH Auto-Fix (funktioniert immer!)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Warnings unterdrücken
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from envs.economy_env import EconomyEnv


def train_government(num_iterations=20, max_years=5):
    """
    Trainiere Regierungs-Agent
    
    Args:
        num_iterations: Anzahl Training-Iterationen (nicht Jahre!)
        max_years: Jahre pro Episode
    """
    print("\n" + "="*60)
    print("🧠 VWL-RL Training: Regierungs-Agent (Single-Agent)")
    print("="*60)
    print(f"Zeitstruktur: {max_years} Jahre/Episode = {365*max_years} Steps")
    print(f"Training: {num_iterations} Iterationen")
    print("="*60 + "\n")
    
    # Ray initialisieren (mit log_level=ERROR für cleane Ausgabe)
    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        log_to_driver=False  # Keine Worker-Logs im Driver
    )
    
    # PPO Config (Ray 2.10 API!)
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
            log_level="ERROR"  # Nur Fehler, keine Warnings
        )
    )
    
    # Algo erstellen
    algo = config.build()
    
    print("🚀 Training startet...\n")
    
    # Training Loop
    best_reward = float('-inf')
    checkpoint_path = None
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Nur wichtigste Metriken extrahieren
        episode_reward_mean = result.get('episode_reward_mean', 0)
        episode_len_mean = result.get('episode_len_mean', 0)
        
        # Kurze, klare Ausgabe
        print(f"📊 Iteration {i+1:2d}/{num_iterations} | "
              f"Reward: {episode_reward_mean:8.2f} | "
              f"Years: {episode_len_mean/365:.1f}")
        
        # Checkpoint bei Verbesserung
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            checkpoint_path = algo.save()
            print(f"   ✅ Neuer Bestwert!")
        
        # Alle 5 Iterationen: Checkpoint
        if (i + 1) % 5 == 0:
            checkpoint_path = algo.save()
            print(f"   💾 Checkpoint gespeichert\n")
    
    # Final Checkpoint
    final_checkpoint = algo.save()
    
    print("\n" + "="*60)
    print("✅ Training abgeschlossen!")
    print(f"Bester Reward: {best_reward:.2f}")
    print(f"Model gespeichert unter: {final_checkpoint}")
    print("="*60 + "\n")
    
    # Aufräumen
    algo.stop()
    ray.shutdown()
    
    return final_checkpoint


if __name__ == "__main__":
    # Training konfigurieren
    NUM_ITERATIONS = 20
    MAX_YEARS = 5
    
    checkpoint = train_government(
        num_iterations=NUM_ITERATIONS,
        max_years=MAX_YEARS
    )
    
    print(f"\n🎯 Nächste Schritte:")
    print(f"1. Model evaluieren: python tests/test_scenarios.py")
    print(f"2. Frontend testen: streamlit run frontend/app.py")
    print(f"3. Model laden für Inference\n")
