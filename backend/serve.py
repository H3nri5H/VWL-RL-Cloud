"""FastAPI Backend (Zustandsbehaftet) - RL Model Inference"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Optional
import os

# TODO: Uncomment when training done
# import ray
# from ray.rllib.algorithms.ppo import PPO

app = FastAPI(
    title="VWL-RL Backend",
    description="Zustandsbehaftete RL-Inference API",
    version="1.0.0"
)

# Global state: Loaded RL model
model = None
model_loaded = False


class SimulationRequest(BaseModel):
    """Request für Simulation"""
    tax_rate: float = 0.3
    gov_spending: float = 500.0
    interest_rate: float = 0.05
    scenario: str = "Normal"
    num_steps: int = 100
    use_rl: bool = False


class SimulationResponse(BaseModel):
    """Response mit Simulations-Ergebnissen"""
    bip: List[float]
    inflation: List[float]
    unemployment: List[float]
    avg_wage: List[float]
    steps: List[int]
    scenario: str
    used_rl: bool


@app.on_event("startup")
async def load_model():
    """Lade RL Model beim Server-Start (zustandsbehaftet!)"""
    global model, model_loaded
    
    model_path = os.getenv("MODEL_PATH", "models/checkpoint_000050")
    
    if os.path.exists(model_path):
        try:
            # TODO: Uncomment when model trained
            # ray.init(ignore_reinit_error=True)
            # model = PPO.from_checkpoint(model_path)
            # model_loaded = True
            print(f"✅ RL Model geladen: {model_path}")
        except Exception as e:
            print(f"⚠️ Model laden fehlgeschlagen: {e}")
            model_loaded = False
    else:
        print(f"⚠️ Kein Model gefunden unter {model_path}")
        model_loaded = False


@app.get("/")
async def root():
    return {
        "service": "VWL-RL Backend",
        "status": "running",
        "model_loaded": model_loaded
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_loaded}


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """
    Führe Wirtschafts-Simulation aus
    
    - Wenn use_rl=True: RL-Agent steuert (zustandsbehaftet!)
    - Sonst: Manuelle Parameter
    """
    
    # Mock Simulation (bis RL-Model trainiert)
    steps = list(range(request.num_steps))
    
    # Szenario-basierte Simulation
    if request.scenario == "Rezession":
        bip = [1000 - i * 5 + np.random.normal(0, 20) for i in steps]
        unemployment = [0.05 + i * 0.003 + np.random.normal(0, 0.01) for i in steps]
        inflation = [0.02 - i * 0.0001 + np.random.normal(0, 0.005) for i in steps]
    elif request.scenario == "Boom":
        bip = [1000 + i * 10 + np.random.normal(0, 30) for i in steps]
        unemployment = [max(0.01, 0.05 - i * 0.0002 + np.random.normal(0, 0.005)) for i in steps]
        inflation = [0.02 + i * 0.0005 + np.random.normal(0, 0.01) for i in steps]
    elif request.scenario == "Inflation":
        bip = [1000 + i * 2 + np.random.normal(0, 25) for i in steps]
        unemployment = [0.05 + np.random.normal(0, 0.01) for i in steps]
        inflation = [0.02 + i * 0.002 + np.random.normal(0, 0.015) for i in steps]
    else:  # Normal
        bip = [1000 + i * 3 + np.random.normal(0, 20) for i in steps]
        unemployment = [0.05 + np.random.normal(0, 0.01) for i in steps]
        inflation = [0.02 + np.random.normal(0, 0.005) for i in steps]
    
    # Clip values
    bip = [max(100, min(5000, x)) for x in bip]
    unemployment = [max(0, min(0.5, x)) for x in unemployment]
    inflation = [max(-0.1, min(0.3, x)) for x in inflation]
    
    # Mock wage data
    avg_wage = [50 + i * 0.1 + np.random.normal(0, 2) for i in steps]
    
    # TODO: Replace with actual RL inference
    # if request.use_rl and model_loaded:
    #     env = EconomyEnv()
    #     obs = env.reset()
    #     for step in range(request.num_steps):
    #         action = model.compute_single_action(obs)
    #         obs, reward, done, info = env.step(action)
    #         bip.append(info['bip'])
    #         ...
    
    return SimulationResponse(
        bip=bip,
        inflation=inflation,
        unemployment=unemployment,
        avg_wage=avg_wage,
        steps=steps,
        scenario=request.scenario,
        used_rl=request.use_rl and model_loaded
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
