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

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from envs.economy_env import EconomyEnv
import os


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
    
    # Ray initialisieren
    ray.init(ignore_reinit_error=True, num_cpus=4)
    
    # PPO Config (Ray 2.10 API!)
    config = (
        PPOConfig()
        .environment(env=EconomyEnv, env_config={'max_years': max_years})
        .framework("torch")
        .rollouts(                              # Ray 2.10: rollouts() nicht env_runners()!
            num_rollout_workers=2,              # Parallel workers
            rollout_fragment_length="auto"       # Auto-batch
        )
        .training(
            train_batch_size=4000,               # Samples pro Training
            sgd_minibatch_size=256,              # Mini-batch size
            num_sgd_iter=10,                     # SGD iterations
            lr=3e-4,                             # Learning rate
            gamma=0.99,                          # Discount factor
            lambda_=0.95,                        # GAE lambda
            clip_param=0.2,                      # PPO clip
            entropy_coeff=0.01,                  # Exploration
            vf_loss_coeff=0.5                    # Value function loss weight
        )
        .resources(
            num_gpus=0  # CPU training (GPU optional)
        )
        .reporting(
            min_sample_timesteps_per_iteration=4000
        )
    )
    
    # Algo erstellen
    algo = config.build()
    
    print("🚀 Training startet...\n")
    
    # Training Loop
    best_reward = float('-inf')
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Metriken extrahieren (Ray 2.10 Format)
        episode_reward_mean = result.get('episode_reward_mean', 0)
        episode_len_mean = result.get('episode_len_mean', 0)
        
        print(f"\n📊 Iteration {i+1}/{num_iterations}")
        print(f"   Episode Reward Mean: {episode_reward_mean:.2f}")
        print(f"   Episode Length Mean: {episode_len_mean:.1f} Steps")
        print(f"   Entspricht: {episode_len_mean/365:.2f} Jahren")
        
        # Checkpoint bei Verbesserung
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            checkpoint_dir = algo.save()
            print(f"   ✅ Neuer Bestwert! Checkpoint: {checkpoint_dir}")
        
        # Alle 5 Iterationen: Checkpoint
        if (i + 1) % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"   💾 Checkpoint gespeichert: {checkpoint_dir}")
    
    # Final Checkpoint
    final_checkpoint = algo.save()
    
    print("\n" + "="*60)
    print("✅ Training abgeschlossen!")
    print(f"Bester Reward: {best_reward:.2f}")
    print(f"Final Checkpoint: {final_checkpoint}")
    print("="*60 + "\n")
    
    # Aufräumen
    algo.stop()
    ray.shutdown()
    
    return final_checkpoint


if __name__ == "__main__":
    # Training konfigurieren
    NUM_ITERATIONS = 20      # Training iterations (nicht Jahre!)
    MAX_YEARS = 5            # Jahre pro Episode
    
    checkpoint = train_government(
        num_iterations=NUM_ITERATIONS,
        max_years=MAX_YEARS
    )
    
    print(f"\n🎯 Nächste Schritte:")
    print(f"1. Model laden: algo = Algorithm.from_checkpoint('{checkpoint}')")
    print(f"2. Evaluieren: python tests/test_scenarios.py")
    print(f"3. Frontend testen: streamlit run frontend/app.py\n")
