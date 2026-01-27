"""Cloud Training Script für Ray RLlib

Trainiert Model in Cloud und speichert Checkpoint in Google Cloud Storage.
Wird als Cloud Run Job ausgeführt.

Environment Variables:
- NUM_ITERATIONS: Anzahl Trainings-Iterationen (default: 100)
- MAX_YEARS: Jahre pro Episode (default: 10)
- GCS_BUCKET: GCS Bucket Name (z.B. vwl-rl-models)
"""

import os
import sys
from pathlib import Path
import shutil

# PYTHONPATH Fix
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from envs.economy_env import EconomyEnv

# Google Cloud Storage
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    print("⚠️  google-cloud-storage nicht installiert")
    GCS_AVAILABLE = False


def upload_to_gcs(local_dir: str, bucket_name: str, gcs_path: str):
    """Upload Checkpoint zu Google Cloud Storage
    
    Args:
        local_dir: Lokaler Checkpoint-Pfad
        bucket_name: GCS Bucket Name
        gcs_path: Ziel-Pfad im Bucket
    """
    if not GCS_AVAILABLE:
        print("❌ GCS Upload übersprungen (google-cloud-storage fehlt)")
        return
    
    print(f"\n📤 Upload zu GCS: gs://{bucket_name}/{gcs_path}")
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Upload alle Dateien im Checkpoint-Ordner
        local_path = Path(local_dir)
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                # Relativer Pfad für GCS
                relative_path = file_path.relative_to(local_path)
                blob_path = f"{gcs_path}/{relative_path}"
                
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(file_path))
                print(f"  ✅ {relative_path}")
        
        print(f"\n✅ Upload erfolgreich: gs://{bucket_name}/{gcs_path}")
        
    except Exception as e:
        print(f"\n❌ Upload Fehler: {e}")
        raise


def train_cloud(num_iterations: int, max_years: int, gcs_bucket: str = None):
    """Training in der Cloud
    
    Args:
        num_iterations: Anzahl Trainings-Iterationen
        max_years: Jahre pro Episode
        gcs_bucket: GCS Bucket Name (optional)
    """
    print("\n" + "="*60)
    print("☁️  VWL-RL Cloud Training")
    print("="*60)
    print(f"Iterationen: {num_iterations}")
    print(f"Jahre/Episode: {max_years}")
    print(f"GCS Bucket: {gcs_bucket or 'Nicht konfiguriert'}")
    print("="*60 + "\n")
    
    # Ray initialisieren
    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        logging_level="ERROR"
    )
    print("✅ Ray initialized\n")
    
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
    print("🚀 Training startet...\n")
    
    # Training Loop
    best_reward = float('-inf')
    best_checkpoint = None
    
    for i in range(num_iterations):
        result = algo.train()
        
        episode_reward_mean = result.get('episode_reward_mean', 0)
        total_iterations = result.get('training_iteration', i+1)
        
        print(f"📊 Iteration {total_iterations:3d}/{num_iterations} | "
              f"Reward: {episode_reward_mean:9.2f}")
        
        # Bestes Model speichern
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            # Temporärer Checkpoint
            checkpoint_path = Path("/tmp/checkpoints") / f"checkpoint_{total_iterations:06d}"
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            algo.save(checkpoint_dir=str(checkpoint_path))
            best_checkpoint = str(checkpoint_path)
            print(f"   ✅ Neuer Bestwert!")
        
        # Progress-Report alle 10 Iterationen
        if total_iterations % 10 == 0:
            print(f"\n📈 Fortschritt: {total_iterations}/{num_iterations} ({total_iterations/num_iterations*100:.0f}%)")
            print(f"   Bester Reward: {best_reward:.2f}\n")
    
    # Final Checkpoint
    final_path = Path("/tmp/checkpoints/checkpoint_final")
    final_path.parent.mkdir(exist_ok=True, parents=True)
    algo.save(checkpoint_dir=str(final_path))
    
    print("\n" + "="*60)
    print("✅ Training abgeschlossen!")
    print(f"Bester Reward: {best_reward:.2f}")
    print(f"Final Checkpoint: {final_path}")
    print("="*60 + "\n")
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    
    # Upload zu GCS
    if gcs_bucket and GCS_AVAILABLE:
        print("\n📤 Uploading zu Google Cloud Storage...")
        upload_to_gcs(
            local_dir=str(final_path),
            bucket_name=gcs_bucket,
            gcs_path="models/checkpoint_final"
        )
        
        if best_checkpoint:
            upload_to_gcs(
                local_dir=best_checkpoint,
                bucket_name=gcs_bucket,
                gcs_path="models/checkpoint_best"
            )
    else:
        print("⚠️  GCS Upload übersprungen (kein Bucket konfiguriert)")
    
    print("\n✅ Cloud Training fertig!\n")
    return str(final_path)


if __name__ == "__main__":
    # Config aus Environment Variables
    NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS", "100"))
    MAX_YEARS = int(os.getenv("MAX_YEARS", "10"))
    GCS_BUCKET = os.getenv("GCS_BUCKET", "")
    
    # Training starten
    train_cloud(
        num_iterations=NUM_ITERATIONS,
        max_years=MAX_YEARS,
        gcs_bucket=GCS_BUCKET if GCS_BUCKET else None
    )
