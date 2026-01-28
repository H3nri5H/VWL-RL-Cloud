"""Lokales Training für Multi-Agent Wirtschafts-Simulation

Usage:
    python train/train_local.py --version v1.0
    python train/train_local.py --version v1.0 --timesteps 500000
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Import custom environment
from envs.multi_agent_economy import MultiAgentEconomyEnv


def make_env():
    """Create environment instance"""
    return MultiAgentEconomyEnv(config_path='configs')


def train(version, timesteps, learning_rate):
    """
    Train Multi-Agent RL Model
    
    Args:
        version: Model version (e.g., 'v1.0')
        timesteps: Total training steps
        learning_rate: PPO learning rate
    """
    print("="*60)
    print(f"🚀 VWL Multi-Agent RL Training")
    print("="*60)
    print(f"Version:        {version}")
    print(f"Timesteps:      {timesteps:,}")
    print(f"Episodes:       {timesteps // 250:,}")
    print(f"Learning Rate:  {learning_rate}")
    print("="*60)
    print()
    
    # Load training config
    with open('configs/training_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    print("🏭 Creating environment...")
    env = make_env()
    
    # Wrap für Stable-Baselines3 (braucht vectorized env)
    # TODO: Später mit echtem Multi-Agent Wrapper
    # Für jetzt: Trainiere nur EINEN Agent (Haushalt_0) als Demo
    print("⚠️  DEMO: Trainiere nur Haushalt_0 (Single-Agent Modus)")
    print("    Multi-Agent Training wird später implementiert.\n")
    
    # TODO: Hier Single-Agent Wrapper einbauen
    # env = SingleAgentWrapper(env, agent_id='household_0')
    # env = DummyVecEnv([lambda: env])
    
    # Für jetzt: Skip Training, nur Environment testen
    print("🧪 Testing environment...")
    obs, info = env.reset()
    print(f"   Initial observations: {len(obs)} agents")
    print(f"   Haushalte: {info['active_households']}")
    print(f"   Firmen: {info['active_firms']}")
    print()
    
    # Test 10 Steps
    print("📍 Running 10 test steps...")
    for step in range(10):
        # Random actions für alle Agents
        actions = {}
        for i in range(10):
            actions[f'household_{i}'] = env.household_action_space.sample()
        for i in range(5):
            actions[f'firm_{i}'] = env.firm_action_space.sample()
        
        obs, rewards, done, truncated, info = env.step(actions)
        
        if step == 0:
            print(f"   Step 1: {info['active_households']} Haushalte, {info['active_firms']} Firmen aktiv")
    
    print()
    print("✅ Environment funktioniert!\n")
    
    # PPO Training (kommt später)
    print("🚧 PPO Training wird später implementiert.")
    print("   Nächste Schritte:")
    print("   1. Single-Agent Wrapper erstellen")
    print("   2. Multi-Agent Training Framework (z.B. RLlib)")
    print("   3. Separate Policies pro Agent-Typ\n")
    
    # Erstelle Output-Verzeichnisse
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    print("💾 Ordner erstellt: models/, logs/, checkpoints/")
    print()
    print("="*60)
    print("✅ Setup komplett! Bereit für Multi-Agent Training.")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VWL Multi-Agent RL')
    parser.add_argument('--version', type=str, required=True, 
                        help='Model version (e.g., v1.0)')
    parser.add_argument('--timesteps', type=int, default=1_250_000,
                        help='Total training timesteps (default: 1.25M = 5000 episodes)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    
    args = parser.parse_args()
    
    train(args.version, args.timesteps, args.lr)
