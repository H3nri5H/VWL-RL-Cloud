"""Single-Agent Training: 1 Regierungs-Agent lernt Wirtschaftspolitik"""
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from envs.economy_env import EconomyEnv
import os


def main():
    # Ray initialisieren
    ray.init(ignore_reinit_error=True)
    
    print("\n🚀 Starting Single-Agent Training: Regierungs-RL-Agent")
    print("="*60)
    
    # PPO Config
    config = (
        PPOConfig()
        .environment(EconomyEnv, env_config={
            "num_firms": 10,
            "num_households": 50
        })
        .framework("torch")
        .env_runners(num_env_runners=2)
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
        )
        .resources(
            num_gpus=0,  # CPU only
        )
    )
    
    # Algo erstellen
    algo = config.build()
    
    print("\n✅ Environment & PPO erstellt")
    print(f"   - Observation Space: {algo.env_creator(None).observation_space}")
    print(f"   - Action Space: {algo.env_creator(None).action_space}")
    
    # Training Loop
    num_iterations = 50
    print(f"\n🏋️ Training für {num_iterations} Iterationen...\n")
    
    best_reward = -float('inf')
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Metrics
        ep_reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0)
        ep_len_mean = result.get("env_runners", {}).get("episode_len_mean", 0)
        
        print(f"Iteration {i+1}/{num_iterations} | "
              f"Reward: {ep_reward_mean:.2f} | "
              f"Ep Length: {ep_len_mean:.1f}")
        
        # Best Model speichern
        if ep_reward_mean > best_reward:
            best_reward = ep_reward_mean
            checkpoint_dir = algo.save("models")
            print(f"   💾 Neues Best Model gespeichert: {checkpoint_dir}")
    
    print("\n" + "="*60)
    print(f"✅ Training abgeschlossen! Best Reward: {best_reward:.2f}")
    print(f"📁 Model gespeichert in: models/")
    
    # Cleanup
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
