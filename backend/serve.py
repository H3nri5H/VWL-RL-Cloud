"""FastAPI Backend für RL-Model Inference

Zustandsbehaftet: Model wird beim Start geladen und bleibt im RAM.
Endpoints:
- POST /predict: Inference für gegebene Observations
- GET /health: Health Check
- GET /info: Model-Informationen
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# PYTHONPATH Fix
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO

# Environment Import
from envs.economy_env import EconomyEnv


# === PYDANTIC MODELS ===

class ObservationInput(BaseModel):
    """Input: Observation State"""
    bip: float
    inflation: float
    unemployment: float
    debt: float
    interest_rate: float

class ActionOutput(BaseModel):
    """Output: Predicted Action"""
    tax_rate: float
    gov_spending: float
    interest_rate: float

class HealthResponse(BaseModel):
    """Health Check Response"""
    status: str
    model_loaded: bool
    ray_initialized: bool

class InfoResponse(BaseModel):
    """Model Info Response"""
    model_type: str
    framework: str
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]


# === FASTAPI APP ===

app = FastAPI(
    title="VWL-RL Inference API",
    description="Reinforcement Learning Model Inference für Wirtschaftspolitik",
    version="1.0.0"
)

# Global State
model = None
model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load Model beim Start"""
    global model, model_loaded
    
    print("🚀 Starting Backend...")
    
    # Ray initialisieren
    ray.init(
        ignore_reinit_error=True,
        num_cpus=2,
        logging_level="ERROR"
    )
    print("✅ Ray initialized")
    
    # Model-Pfad (aus Environment oder default)
    model_path = os.getenv("MODEL_PATH", "checkpoints/checkpoint_final")
    
    if Path(model_path).exists():
        print(f"📥 Loading model from: {model_path}")
        try:
            model = PPO.from_checkpoint(model_path)
            model_loaded = True
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Backend startet ohne Model (nur Health Check verfügbar)")
    else:
        print(f"⚠️  Model nicht gefunden: {model_path}")
        print("⚠️  Backend startet ohne Model (nur Health Check verfügbar)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup beim Herunterfahren"""
    global model
    
    print("🛝 Shutting down...")
    
    if model is not None:
        model.stop()
    
    ray.shutdown()
    print("✅ Shutdown complete")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root Endpoint"""
    return {
        "service": "VWL-RL Inference API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/info", "/predict"]
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health Check für Cloud Run"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        ray_initialized=ray.is_initialized()
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    """Model Informationen"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return InfoResponse(
        model_type="PPO",
        framework="torch",
        observation_space={
            "type": "Box",
            "shape": [5],
            "low": [-10.0, -0.5, 0.0, -10.0, 0.0],
            "high": [10.0, 0.5, 1.0, 10.0, 0.3]
        },
        action_space={
            "type": "Box",
            "shape": [3],
            "low": [0.0, 0.0, 0.0],
            "high": [0.5, 1000.0, 0.2]
        }
    )


@app.post("/predict", response_model=ActionOutput)
async def predict(obs: ObservationInput):
    """Model Inference
    
    Args:
        obs: Observation State [bip, inflation, unemployment, debt, interest_rate]
        
    Returns:
        action: [tax_rate, gov_spending, interest_rate]
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Observation zu numpy array
        observation = np.array([
            obs.bip,
            obs.inflation,
            obs.unemployment,
            obs.debt,
            obs.interest_rate
        ], dtype=np.float32)
        
        # Inference
        action = model.compute_single_action(observation)
        
        # Output
        return ActionOutput(
            tax_rate=float(action[0]),
            gov_spending=float(action[1]),
            interest_rate=float(action[2])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
