"""FastAPI Backend für RL-Model Inference

Zustandsbehaftet: Model und Environment werden geladen und bleiben im RAM.
Endpoints:
- POST /simulate: Vollständige Simulation über N Steps
- POST /predict: Single-Step Inference
- GET /health: Health Check
"""

import os
import sys
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Environment Import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from envs.economy_env import EconomyEnv
except ImportError:
    EconomyEnv = None
    print("⚠️ Warning: Could not import EconomyEnv")


# === PYDANTIC MODELS ===

class SimulationRequest(BaseModel):
    """Request für vollständige Simulation"""
    environment: str = "FullEconomy-v0"
    num_steps: int = 100
    scenario: str = "Normal"
    use_rl_agent: bool = False
    manual_params: Optional[Dict[str, float]] = None

class SimulationStep(BaseModel):
    """Ein Step der Simulation"""
    step: int
    bip: float
    inflation: float
    unemployment: float
    debt: float
    tax_rate: float
    gov_spending: float
    interest_rate: float

class SimulationResponse(BaseModel):
    """Response mit allen Simulation Steps"""
    steps: List[SimulationStep]
    summary: Dict[str, float]

class ObservationInput(BaseModel):
    """Input: Observation State für Single Prediction"""
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
    env_available: bool
    model_loaded: bool


# === FASTAPI APP ===

app = FastAPI(
    title="VWL-RL Backend API",
    description="Backend für Volkswirtschafts-Simulation mit Reinforcement Learning",
    version="2.0.0"
)

# CORS für Frontend-Kommunikation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State (zustandsbehaftet!)
model = None
model_loaded = False


def load_model_lazy():
    """Lazy Loading: Model erst beim ersten Request laden"""
    global model, model_loaded
    
    if model_loaded:
        return
    
    print("📥 Attempting to load RL model...")
    
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPO
        
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            logging_level="ERROR"
        )
        
        model_path = os.getenv("MODEL_PATH", "checkpoints/checkpoint_final")
        
        if os.path.exists(model_path):
            model = PPO.from_checkpoint(model_path)
            model_loaded = True
            print("✅ Model loaded successfully!")
        else:
            print(f"⚠️ Model not found at: {model_path}")
            print("   Continuing without RL model (manual mode only)")
            
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        print("   Continuing without RL model (manual mode only)")


def create_environment(env_name: str, scenario: str):
    """Erstelle Environment Instanz"""
    if EconomyEnv is None:
        raise HTTPException(status_code=503, detail="Environment not available")
    
    try:
        env = EconomyEnv()
        return env
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Environment creation failed: {str(e)}")


def get_manual_action(params: Dict[str, float]) -> np.ndarray:
    """Erstelle Action aus manuellen Parametern"""
    return np.array([
        params.get('tax_rate', 0.3),
        params.get('gov_spending', 500.0),
        params.get('interest_rate', 0.05)
    ], dtype=np.float32)


def get_rl_action(observation: np.ndarray) -> np.ndarray:
    """Hole Action vom RL-Model"""
    global model, model_loaded
    
    if not model_loaded:
        load_model_lazy()
    
    if not model_loaded:
        # Fallback: Einfache Regel-basierte Policy
        return np.array([0.3, 500.0, 0.05], dtype=np.float32)
    
    try:
        action = model.compute_single_action(observation)
        return np.array(action, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Model inference failed: {e}")
        # Fallback
        return np.array([0.3, 500.0, 0.05], dtype=np.float32)


@app.on_event("startup")
async def startup_event():
    """Startup Event"""
    print("🚀 VWL-RL Backend started")
    print(f"   Environment available: {EconomyEnv is not None}")


@app.get("/")
async def root():
    """Root Endpoint"""
    return {
        "service": "VWL-RL Backend API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": ["/simulate", "/predict", "/health"]
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health Check"""
    return HealthResponse(
        status="healthy",
        env_available=EconomyEnv is not None,
        model_loaded=model_loaded
    )


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """Führe vollständige Simulation aus"""
    
    # Environment erstellen
    env = create_environment(request.environment, request.scenario)
    
    # Reset
    obs, info = env.reset()
    
    # Simulation durchführen
    steps_data = []
    
    for step in range(request.num_steps):
        # Action bestimmen
        if request.use_rl_agent:
            action = get_rl_action(obs)
        else:
            action = get_manual_action(request.manual_params or {})
        
        # Environment Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Daten sammeln
        step_data = SimulationStep(
            step=step,
            bip=float(info['bip']),
            inflation=float(info['inflation']),
            unemployment=float(info['unemployment']),
            debt=float(info['debt']),
            tax_rate=float(action[0]),
            gov_spending=float(action[1]),
            interest_rate=float(action[2])
        )
        steps_data.append(step_data)
        
        if terminated or truncated:
            break
    
    # Summary berechnen
    bip_values = [s.bip for s in steps_data]
    inflation_values = [s.inflation for s in steps_data]
    unemployment_values = [s.unemployment for s in steps_data]
    
    summary = {
        "final_bip": bip_values[-1],
        "bip_growth": ((bip_values[-1] / bip_values[0]) - 1) * 100 if bip_values[0] > 0 else 0,
        "avg_inflation": float(np.mean(inflation_values)) * 100,
        "avg_unemployment": float(np.mean(unemployment_values)) * 100,
        "final_debt": steps_data[-1].debt
    }
    
    return SimulationResponse(
        steps=steps_data,
        summary=summary
    )


@app.post("/predict", response_model=ActionOutput)
async def predict(obs: ObservationInput):
    """Single-Step Prediction (für externe Environments)"""
    
    observation = np.array([
        obs.bip,
        obs.inflation,
        obs.unemployment,
        obs.debt,
        obs.interest_rate
    ], dtype=np.float32)
    
    action = get_rl_action(observation)
    
    return ActionOutput(
        tax_rate=float(action[0]),
        gov_spending=float(action[1]),
        interest_rate=float(action[2])
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
