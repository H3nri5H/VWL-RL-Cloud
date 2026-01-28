"""Analyze trained agent behavior and show detailed metrics"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
from ray.rllib.algorithms.ppo import PPO
from envs.rllib_economy_env import RLlibEconomyEnv


def analyze_agents(checkpoint_path, num_episodes=10):
    """Analyze trained agents over multiple episodes"""
    
    print("\n" + "="*60)
    print("Trained Agent Analysis")
    print("="*60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}\n")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, logging_level='ERROR')
    
    # Load trained algorithm
    print("[1/4] Loading checkpoint...")
    algo = PPO.from_checkpoint(checkpoint_path)
    
    # Create environment
    print("[2/4] Creating environment...")
    env = RLlibEconomyEnv()
    
    # Storage for metrics
    metrics = defaultdict(list)
    
    print("[3/4] Running episodes...\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = {"__all__": False}
        episode_reward = 0
        step = 0
        
        # Track episode-specific metrics
        episode_prices = []
        episode_wages = []
        episode_production = []
        episode_consumption = []
        
        while not done["__all__"] and step < 1250:
            # Get actions from trained policies
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = 'household_policy' if 'household' in agent_id else 'firm_policy'
                actions[agent_id] = algo.compute_single_action(agent_obs, policy_id=policy_id)
            
            # Step environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Extract metrics from this step
            for agent_id, action in actions.items():
                if 'firm' in agent_id:
                    # Firm action: [production_factor, wage_factor, price_factor]
                    episode_production.append(action[0])
                    episode_wages.append(action[1])
                    episode_prices.append(action[2])
                else:
                    # Household action: [consumption_rate]
                    episode_consumption.append(action[0])
            
            episode_reward += sum(rewards.values())
            done = terminateds
            done["__all__"] = all(terminateds.values()) or all(truncateds.values())
            step += 1
        
        # Store episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step)
        metrics['avg_prices'].append(np.mean(episode_prices) if episode_prices else 0)
        metrics['avg_wages'].append(np.mean(episode_wages) if episode_wages else 0)
        metrics['avg_production'].append(np.mean(episode_production) if episode_production else 0)
        metrics['avg_consumption'].append(np.mean(episode_consumption) if episode_consumption else 0)
        
        # Get final info
        final_info = list(infos.values())[0] if infos else {}
        metrics['bankrupt_firms'].append(final_info.get('bankrupt_firms', 0))
        metrics['bankrupt_households'].append(final_info.get('bankrupt_households', 0))
        
        print(f"Episode {episode+1}/{num_episodes}: Reward={episode_reward:.0f}, Steps={step}, Bankruptcies=F:{metrics['bankrupt_firms'][-1]} H:{metrics['bankrupt_households'][-1]}")
    
    print("\n[4/4] Computing statistics...\n")
    
    # Print results
    print("="*60)
    print("RESULTS")
    print("="*60)
    
    print("\n📊 Episode Statistics:")
    print(f"  Avg Reward: {np.mean(metrics['episode_rewards']):.2f} ± {np.std(metrics['episode_rewards']):.2f}")
    print(f"  Avg Length: {np.mean(metrics['episode_lengths']):.1f} steps")
    
    print("\n🏭 Firm Behavior:")
    print(f"  Avg Price Factor: {np.mean(metrics['avg_prices']):.3f} ± {np.std(metrics['avg_prices']):.3f}")
    print(f"  Avg Wage Factor: {np.mean(metrics['avg_wages']):.3f} ± {np.std(metrics['avg_wages']):.3f}")
    print(f"  Avg Production Factor: {np.mean(metrics['avg_production']):.3f} ± {np.std(metrics['avg_production']):.3f}")
    print(f"  Bankruptcies: {np.mean(metrics['bankrupt_firms']):.2f} per episode")
    
    print("\n🏠 Household Behavior:")
    print(f"  Avg Consumption Rate: {np.mean(metrics['avg_consumption']):.3f} ± {np.std(metrics['avg_consumption']):.3f}")
    print(f"  Bankruptcies: {np.mean(metrics['bankrupt_households']):.2f} per episode")
    
    print("\n" + "="*60)
    print("\nInterpretation:")
    print("  - Price/Wage/Production Factors are normalized actions")
    print("  - Values close to 0 = conservative strategy")
    print("  - Positive values = aggressive strategy")
    print("  - Low bankruptcies = stable economy!")
    print("\n")
    
    # Cleanup
    ray.shutdown()
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trained agents")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)"
    )
    
    args = parser.parse_args()
    
    analyze_agents(args.checkpoint, args.episodes)
