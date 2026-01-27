"""FastAPI Backend für RL-Model Inference

Zustandsbehaftet: Model wird beim ersten Request geladen.
Endpoints:
- POST /predict: Inference für gegebene Observations
- GET /health: Health Check
- GET /info: Model-Informationen
"""

import os
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np


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

class InfoResponse(BaseModel):
    """Model Info Response"""
    model_type: str
    framework: str


# === FASTAPI APP ===

app = FastAPI(
    title="VWL-RL Inference API",
    description="Reinforcement Learning Model Inference für Wirtschaftspolitik",
    version="1.0.0"
)

# Global State
model = None
model_loaded = False


def load_model_lazy():
    """Lazy Loading: Model erst beim ersten Request laden"""
    global model, model_loaded
    
    if model_loaded:
        return
    
    print("📥 Loading model (lazy)...")
    
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPO
        
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            logging_level="ERROR"
        )
        print("✅ Ray initialized")
        
        model_path = os.getenv("MODEL_PATH", "checkpoints/checkpoint_final")
        
        if os.path.exists(model_path):
            model = PPO.from_checkpoint(model_path)
            model_loaded = True
            print("✅ Model loaded!")
        else:
            print(f"⚠️ Model not found: {model_path}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")


@app.on_event("startup")
async def startup_event():
    """Fast Startup - kein Model Loading"""
    print("🚀 Backend started (ready for requests)")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root Endpoint"""
    return {
        "service": "VWL-RL Inference API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health Check für Cloud Run"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    """Model Informationen"""
    if not model_loaded:
        load_model_lazy()
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available")
    
    return InfoResponse(
        model_type="PPO",
        framework="torch"
    )


@app.post("/predict", response_model=ActionOutput)
async def predict(obs: ObservationInput):
    """Model Inference"""
    if not model_loaded:
        load_model_lazy()
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        observation = np.array([
            obs.bip,
            obs.inflation,
            obs.unemployment,
            obs.debt,
            obs.interest_rate
        ], dtype=np.float32)
        
        action = model.compute_single_action(observation)
        
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
