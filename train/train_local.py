"""Training Script fuer Multi-Agent Economy mit RLlib

Trainiert alle Agents (Haushalte + Firmen) mit PPO.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from envs.rllib_economy_env import RLlibEconomyEnv


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map agents zu policies
    
    Alle Haushalte teilen sich eine Policy.
    Alle Firmen teilen sich eine Policy.
    """
    if 'household' in agent_id:
        return 'household_policy'
    else:
        return 'firm_policy'


def train_economy(args):
    """Training starten"""
    
    print("="*60)
    print("Multi-Agent Economy Training (RLlib + PPO)")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Checkpoint: {args.checkpoint_freq}")
    print(f"  Output: {args.output_dir}")
    print("\n" + "="*60 + "\n")
    
    # Ray initialisieren
    ray.init(
        ignore_reinit_error=True,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus
    )
    
    # Environment registrieren
    register_env("economy", lambda config: RLlibEconomyEnv(config))
    
    # Dummy Environment fuer Spaces
    dummy_env = RLlibEconomyEnv()
    
    # Multi-Agent Config
    policies = {
        'household_policy': PolicySpec(
            observation_space=dummy_env.household_observation_space,
            action_space=dummy_env.household_action_space,
            config={}
        ),
        'firm_policy': PolicySpec(
            observation_space=dummy_env.firm_observation_space,
            action_space=dummy_env.firm_action_space,
            config={}
        )
    }
    
    # PPO Config
    # IMPORTANT: api_stack() must be called FIRST
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .framework("torch")
        .environment(
            env="economy",
            env_config={
                'config_path': 'configs/agent_config.yaml'
            }
        )
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=['household_policy', 'firm_policy']
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=1
        )
        .resources(
            num_gpus=args.num_gpus
        )
        .debugging(
            log_level="INFO"
        )
    )
    
    # Training starten
    print("[START] Training...\n")
    
    tune.run(
        "PPO",
        name="economy_training",
        config=config.to_dict(),
        stop={
            "timesteps_total": args.timesteps
        },
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=args.output_dir,
        verbose=1
    )
    
    print("\n[SUCCESS] Training abgeschlossen!")
    print(f"Models gespeichert in: {args.output_dir}")
    
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Agent Economy")
    
    # Training Parameter
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps to train (default: 100k)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Checkpoint frequency (default: every 10 iterations)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ray_results",
        help="Output directory for checkpoints (default: ./ray_results)"
    )
    
    # Ressourcen
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of rollout workers (default: 2)"
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs (default: auto-detect)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs (default: 0)"
    )
    
    args = parser.parse_args()
    
    train_economy(args)
