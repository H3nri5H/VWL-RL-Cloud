"""Training Script fuer Multi-Agent Economy mit RLlib

Trainiert alle Agents (Haushalte + Firmen) mit PPO.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

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
    
    print("\n" + "="*60)
    print("Multi-Agent Economy Training (RLlib + PPO)")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Checkpoint Frequency: every {args.checkpoint_freq} iterations")
    print(f"  Output: {args.output_dir}")
    print("\n" + "="*60 + "\n")
    
    # Ray initialisieren - mit weniger Logging
    ray.init(
        ignore_reinit_error=True,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        log_to_driver=False,  # Weniger Logs
        logging_level='ERROR'  # Nur Errors
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
    
    # PPO Config - Ray 2.30.0 API
    config = (
        PPOConfig()
        .environment(
            env="economy",
            env_config={
                'config_path': 'configs/agent_config.yaml'
            }
        )
        .framework("torch")
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
            log_level="ERROR"  # Nur Errors loggen
        )
    )
    
    # Training starten
    print("[START] Training...\n")
    
    # Convert to absolute path for PyArrow
    storage_path = Path(args.output_dir).absolute()
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Custom progress reporter - nur wichtige Metriken
    from ray.tune import CLIReporter
    reporter = CLIReporter(
        max_report_frequency=30,  # Update alle 30 Sekunden
        metric_columns=[
            "training_iteration",
            "timesteps_total",
            "episode_reward_mean",
            "time_total_s"
        ]
    )
    
    # Ray 2.30.0: Use absolute storage_path
    tune.run(
        "PPO",
        name="economy_training",
        config=config.to_dict(),
        stop={
            "timesteps_total": args.timesteps
        },
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        storage_path=str(storage_path),
        verbose=0,  # Keine detaillierten Logs
        progress_reporter=reporter,
        log_to_file=True  # Logs in Files statt Console
    )
    
    print("\n" + "="*60)
    print("[SUCCESS] Training abgeschlossen!")
    print("="*60)
    print(f"\nModels gespeichert in: {storage_path}/economy_training")
    print(f"\nTensorBoard starten:")
    print(f"  tensorboard --logdir {storage_path}/economy_training")
    print("\n")
    
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
