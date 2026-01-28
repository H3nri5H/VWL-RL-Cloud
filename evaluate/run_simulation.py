"""Run economy simulation with latest trained checkpoint

Direct evaluation without version conflicts.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent dir
sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
from ray.rllib.algorithms.ppo import PPO
from envs.rllib_economy_env import RLlibEconomyEnv


def run_simulation(checkpoint_path, steps=250):
    """Run simulation with trained agents"""
    
    print("\n" + "="*60)
    print("Economy Simulation")
    print("="*60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Steps: {steps}\n")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, logging_level='ERROR')
    
    # Load algorithm
    print("[1/3] Loading trained agents...")
    algo = PPO.from_checkpoint(str(Path(checkpoint_path).absolute()))
    
    # Create environment
    print("[2/3] Creating environment...")
    env = RLlibEconomyEnv()
    
    # Run simulation
    print(f"[3/3] Running {steps} steps...\n")
    
    obs, info = env.reset()
    
    # Storage
    history = {
        'step': [],
        'total_reward': [],
        'avg_price': [],
        'avg_wage': [],
        'bankruptcies': []
    }
    
    for step in range(steps):
        # Get actions from policies
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy_id = 'household_policy' if 'household' in agent_id else 'firm_policy'
            actions[agent_id] = algo.compute_single_action(agent_obs, policy_id=policy_id)
        
        # Step environment
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        # Collect metrics
        step_prices = []
        step_wages = []
        
        for agent_id, action in actions.items():
            if 'firm' in agent_id:
                step_prices.append(action[2])
                step_wages.append(action[1])
        
        agent_info = list(infos.values())[0] if infos else {}
        
        history['step'].append(step)
        history['total_reward'].append(sum(rewards.values()))
        history['avg_price'].append(np.mean(step_prices) if step_prices else 0)
        history['avg_wage'].append(np.mean(step_wages) if step_wages else 0)
        history['bankruptcies'].append(
            agent_info.get('bankrupt_firms', 0) + agent_info.get('bankrupt_households', 0)
        )
        
        # Progress
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{steps}: Reward={sum(rewards.values()):.0f}, Bankruptcies={history['bankruptcies'][-1]}")
        
        # Check done
        if all(terminateds.values()) or all(truncateds.values()):
            break
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"\n  Avg Reward: {np.mean(history['total_reward']):.2f}")
    print(f"  Avg Price: {np.mean(history['avg_price']):.3f}")
    print(f"  Avg Wage: {np.mean(history['avg_wage']):.3f}")
    print(f"  Total Bankruptcies: {sum(history['bankruptcies'])}")
    print("\n")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(history['step'], history['total_reward'])
    axes[0, 0].set_title('Total Reward')
    axes[0, 0].set_xlabel('Steps')
    
    axes[0, 1].plot(history['step'], history['avg_price'])
    axes[0, 1].set_title('Average Price')
    axes[0, 1].set_xlabel('Steps')
    
    axes[1, 0].plot(history['step'], history['avg_wage'])
    axes[1, 0].set_title('Average Wage')
    axes[1, 0].set_xlabel('Steps')
    
    axes[1, 1].plot(history['step'], history['bankruptcies'])
    axes[1, 1].set_title('Bankruptcies')
    axes[1, 1].set_xlabel('Steps')
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150)
    print("[SAVED] simulation_results.png\n")
    
    plt.show()
    
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Simulation steps (default: 250)"
    )
    
    args = parser.parse_args()
    
    run_simulation(args.checkpoint, args.steps)
