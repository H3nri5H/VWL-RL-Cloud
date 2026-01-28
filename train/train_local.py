"""Training Script fuer Multi-Agent Economy mit RLlib

Trainiert alle Agents (Haushalte + Firmen) mit PPO.
"""

# Warnings GANZ FRUEH unterdrücken - vor allen imports!
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['RAY_DEDUP_LOGS'] = '0'

import argparse
import sys
import signal
import shutil
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from ray.tune import CLIReporter

from envs.rllib_economy_env import RLlibEconomyEnv

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle CTRL+C gracefully"""
    global shutdown_requested
    print("\n\n" + "="*60)
    print("⚠️  CTRL+C detected - Graceful shutdown initiated...")
    print("="*60)
    print("\nSaving current checkpoint and stopping training...")
    print("You can resume later with --resume flag!\n")
    shutdown_requested = True

def cleanup_old_checkpoints(trial_dir, keep_last_n=1):
    """Keep only the last N checkpoints to save disk space
    
    Args:
        trial_dir: Path to trial directory (e.g., ray_results/.../PPO_economy_xxx)
        keep_last_n: Number of recent checkpoints to keep
    """
    trial_path = Path(trial_dir)
    if not trial_path.exists():
        return
    
    # Find all checkpoint directories
    checkpoints = sorted(
        [d for d in trial_path.glob("checkpoint_*") if d.is_dir()],
        key=lambda x: int(x.name.split("_")[-1])
    )
    
    # Delete all except last N
    if len(checkpoints) > keep_last_n:
        for checkpoint in checkpoints[:-keep_last_n]:
            try:
                shutil.rmtree(checkpoint)
                print(f"🗑️  Removed old checkpoint: {checkpoint.name}")
            except Exception as e:
                print(f"⚠️  Could not remove {checkpoint.name}: {e}")

def find_latest_checkpoint(base_path):
    """Find latest checkpoint for resuming training"""
    base = Path(base_path)
    
    if not base.exists():
        return None
    
    # Look for PPO subdirectories
    ppo_dirs = list(base.glob("PPO_*"))
    if not ppo_dirs:
        return None
    
    # Use first (should be only one)
    ppo_dir = ppo_dirs[0]
    checkpoints = sorted(
        [d for d in ppo_dir.glob("checkpoint_*") if d.is_dir()],
        key=lambda x: int(x.name.split("_")[-1])
    )
    
    if not checkpoints:
        return None
    
    return str(checkpoints[-1].absolute())

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map agents zu policies
    
    Alle Haushalte teilen sich eine Policy.
    Alle Firmen teilen sich eine Policy.
    """
    if 'household' in agent_id:
        return 'household_policy'
    else:
        return 'firm_policy'


class TrainingCallback(tune.Callback):
    """Custom callback for checkpoint cleanup and CTRL+C handling"""
    
    def __init__(self, keep_checkpoints=1):
        self.keep_checkpoints = keep_checkpoints
        self.last_cleanup_iter = 0
    
    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called after each training iteration"""
        global shutdown_requested
        
        # Check if shutdown requested
        if shutdown_requested:
            trial.stop()
            return
        
        # Cleanup old checkpoints periodically (every 5 iterations)
        if iteration - self.last_cleanup_iter >= 5:
            cleanup_old_checkpoints(trial.local_path, self.keep_checkpoints)
            self.last_cleanup_iter = iteration


def train_economy(args):
    """Training starten"""
    
    # Setup CTRL+C handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*60)
    print("Multi-Agent Economy Training (RLlib + PPO)")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Checkpoint Frequency: every {args.checkpoint_freq} iterations")
    print(f"  Keep Checkpoints: {args.keep_checkpoints}")
    print(f"  Output: {args.output_dir}")
    
    # Check for resume
    resume_checkpoint = None
    if args.resume:
        storage_path = Path(args.output_dir).absolute() / "economy_training"
        resume_checkpoint = find_latest_checkpoint(storage_path)
        if resume_checkpoint:
            print(f"\n🔄 RESUMING from: {resume_checkpoint}")
        else:
            print(f"\n⚠️  No checkpoint found to resume from. Starting fresh...")
    
    print("\n💡 Press CTRL+C anytime to stop gracefully and save checkpoint!")
    print("\n" + "="*60 + "\n")
    
    # Ray initialisieren - mit minimalem Logging
    ray.init(
        ignore_reinit_error=True,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        log_to_driver=False,
        logging_level='ERROR'
    )
    
    # Environment registrieren (silent)
    register_env("economy", lambda config: RLlibEconomyEnv(config))
    
    # Dummy Environment fuer Spaces (silent output)
    import io
    import contextlib
    
    with contextlib.redirect_stdout(io.StringIO()):
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
            log_level="ERROR"
        )
    )
    
    # Training starten
    print("[START] Training...\n")
    
    # Convert to absolute path for PyArrow
    storage_path = Path(args.output_dir).absolute()
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Custom progress reporter - LIVE UPDATES!
    reporter = CLIReporter(
        max_report_frequency=10,  # Update alle 10 Sekunden
        metric_columns={
            "training_iteration": "Iter",
            "timesteps_total": "Timesteps",
            "env_runners/episode_return_mean": "Reward",
            "env_runners/episode_len_mean": "Ep.Len",
            "time_total_s": "Time(s)",
            "info/learner/default_policy/learner_stats/policy_loss": "Policy Loss",
        },
        metric='env_runners/episode_return_mean',
        mode='max',
        max_progress_rows=20,
        max_error_rows=1,
        max_column_length=15,
        print_intermediate_tables=True  # Show updates!
    )
    
    print("Tipp: Starte in einem zweiten Terminal: tensorboard --logdir ray_results/economy_training\n")
    print("="*60 + "\n")
    
    # Training Callback for cleanup
    callback = TrainingCallback(keep_checkpoints=args.keep_checkpoints)
    
    try:
        # Ray 2.30.0: Use absolute storage_path
        result = tune.run(
            "PPO",
            name="economy_training",
            config=config.to_dict(),
            stop={
                "timesteps_total": args.timesteps
            },
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            keep_checkpoints_num=args.keep_checkpoints,  # RLlib's built-in cleanup
            storage_path=str(storage_path),
            verbose=1,  # Show progress!
            progress_reporter=reporter,
            log_to_file=True,
            raise_on_failed_trial=False,
            resume=args.resume if resume_checkpoint else False,
            restore=resume_checkpoint,
            callbacks=[callback]
        )
        
        print("\n" + "="*60)
        print("[SUCCESS] Training abgeschlossen!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("⚠️  Training stopped by user (CTRL+C)")
        print("="*60)
        result = None
    
    # Final cleanup - keep only last checkpoint
    if result and result.trials:
        trial = result.trials[0]
        cleanup_old_checkpoints(trial.local_path, args.keep_checkpoints)
        
        print(f"\nModels gespeichert in: {trial.local_path}")
        print(f"\nTensorBoard starten:")
        print(f"  tensorboard --logdir {storage_path}/economy_training")
        print("\n" + "="*60)
        print("\nTraining Stats:")
        
        # Zeige finale Stats - mit sicherem Formatting
        if trial.last_result:
            print(f"  Iterations: {trial.last_result.get('training_iteration', 'N/A')}")
            print(f"  Timesteps: {trial.last_result.get('timesteps_total', 'N/A'):,}")
            
            # Try different reward keys
            reward = None
            for key in ['env_runners/episode_return_mean', 'episode_reward_mean', 'episode_return_mean']:
                reward = trial.last_result.get(key)
                if reward is not None:
                    break
            
            if reward is not None and isinstance(reward, (int, float)):
                print(f"  Episode Reward: {reward:.2f}")
            else:
                print(f"  Episode Reward: N/A")
            
            # Safe formatting for time
            time_total = trial.last_result.get('time_total_s', 0)
            if isinstance(time_total, (int, float)):
                print(f"  Training Time: {time_total:.1f}s")
            else:
                print(f"  Training Time: N/A")
    else:
        # Even if interrupted, show where checkpoint is
        print(f"\n💾 Checkpoint saved in: {storage_path}/economy_training")
        print(f"\n🔄 Resume training with: python train/train_local.py --resume --timesteps {args.timesteps}")
    
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
        "--keep-checkpoints",
        type=int,
        default=1,
        help="Number of recent checkpoints to keep (default: 1, saves disk space)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ray_results",
        help="Output directory for checkpoints (default: ./ray_results)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint"
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
