"""Cloud Training Script with GCS Upload and Pub/Sub Notification

Runs in Cloud Run Job:
1. Train RL model (PPO)
2. Upload to Cloud Storage
3. Publish Pub/Sub event
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from google.cloud import storage, pubsub_v1

# Environment
import sys
sys.path.insert(0, '/app')
from envs.economy_env import EconomyEnv

# Config from environment variables
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC", "")
MODEL_VERSION = os.getenv("MODEL_VERSION", "ppo_v1_10M")
TRAINING_STEPS = int(os.getenv("TRAINING_STEPS", "10000000"))  # 10M

print("🏋️ Starting Cloud Training Job")
print(f"   Model: {MODEL_VERSION}")
print(f"   Steps: {TRAINING_STEPS:,}")
print(f"   GCS: {GCS_BUCKET}")
print(f"   Pub/Sub: {PUBSUB_TOPIC}")

# Initialize Ray
ray.init(num_cpus=8, logging_level="INFO")

# Configure PPO
config = (
    PPOConfig()
    .environment(EconomyEnv)
    .framework("torch")
    .rollouts(num_rollout_workers=4)
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        lr=3e-4,
        gamma=0.99,
        clip_param=0.2
    )
)

# Build Algorithm
print("🔨 Building PPO algorithm...")
algo = config.build()

# Training Loop
print("🚀 Starting training...")
total_steps = 0
checkpoint_interval = TRAINING_STEPS // 10  # 10 checkpoints

while total_steps < TRAINING_STEPS:
    result = algo.train()
    total_steps = result['timesteps_total']
    
    if total_steps % checkpoint_interval == 0:
        print(f"📊 Step {total_steps:,} / {TRAINING_STEPS:,}")
        print(f"   Reward: {result['episode_reward_mean']:.2f}")

print("✅ Training complete!")

# Save Checkpoint
checkpoint_path = f"/tmp/{MODEL_VERSION}"
print(f"💾 Saving checkpoint to {checkpoint_path}")
algo.save(checkpoint_path)

# Upload to Cloud Storage
if GCS_BUCKET:
    print(f"📤 Uploading to gs://{GCS_BUCKET}/{MODEL_VERSION}.zip")
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"{MODEL_VERSION}.zip")
    
    # Create zip of checkpoint
    import shutil
    shutil.make_archive(f"/tmp/{MODEL_VERSION}", 'zip', checkpoint_path)
    blob.upload_from_filename(f"/tmp/{MODEL_VERSION}.zip")
    print("✅ Upload complete!")

# Publish Pub/Sub Event
if PUBSUB_TOPIC:
    print(f"📢 Publishing training completion event to {PUBSUB_TOPIC}")
    publisher = pubsub_v1.PublisherClient()
    
    message = json.dumps({
        "event": "training_complete",
        "model_version": MODEL_VERSION,
        "training_steps": total_steps,
        "gcs_path": f"gs://{GCS_BUCKET}/{MODEL_VERSION}.zip"
    })
    
    future = publisher.publish(PUBSUB_TOPIC, message.encode('utf-8'))
    future.result()
    print("✅ Event published!")

print("🎉 Training job completed successfully!")
ray.shutdown()
