"""Test-Script fuer SimpleEconomyEnv

Fuehrt einen kurzen Test durch:
1. Environment initialisieren
2. Prueft ob initiale Bedingungen fix bleiben
3. Prueft ob Seeds funktionieren (Reproduzierbarkeit)
4. Simuliert ein paar Steps
"""

import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.simple_economy_env import SimpleEconomyEnv


def test_fixed_initial_conditions():
    """Test: Initiale Bedingungen bleiben ueber Episoden fix"""
    print("[TEST 1] Fixe Startbedingungen\n")
    
    # Mit festem Seed
    env = SimpleEconomyEnv(seed=42)
    
    # Episode 1
    obs1, info1 = env.reset()
    household_cash_ep1 = [h['cash'] for h in env.households]
    firm_capital_ep1 = [f['capital'] for f in env.firms]
    
    # Ein paar Steps
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    
    # Episode 2 - SOLLTE gleiche Startbedingungen haben!
    obs2, info2 = env.reset()
    household_cash_ep2 = [h['cash'] for h in env.households]
    firm_capital_ep2 = [f['capital'] for f in env.firms]
    
    # Vergleich
    assert household_cash_ep1 == household_cash_ep2, "Haushalte haben unterschiedliche Startbedingungen!"
    assert firm_capital_ep1 == firm_capital_ep2, "Firmen haben unterschiedliche Startbedingungen!"
    
    print("[OK] Startbedingungen bleiben ueber Episoden fix!")
    print(f"     Haushalt 0: Immer {household_cash_ep1[0]:.2f} EUR")
    print(f"     Firma 0: Immer {firm_capital_ep1[0]:.2f} EUR\n")


def test_seed_reproducibility():
    """Test: Gleicher Seed = Gleiche Startbedingungen"""
    print("[TEST 2] Seed Reproduzierbarkeit\n")
    
    # Environment 1 mit Seed 42
    env1 = SimpleEconomyEnv(seed=42)
    env1.reset()
    cash_env1 = [h['cash'] for h in env1.households]
    capital_env1 = [f['capital'] for f in env1.firms]
    
    # Environment 2 mit gleichem Seed 42
    env2 = SimpleEconomyEnv(seed=42)
    env2.reset()
    cash_env2 = [h['cash'] for h in env2.households]
    capital_env2 = [f['capital'] for f in env2.firms]
    
    # Vergleich
    assert cash_env1 == cash_env2, "Haushalte haben unterschiedliche Werte trotz gleichem Seed!"
    assert capital_env1 == capital_env2, "Firmen haben unterschiedliche Werte trotz gleichem Seed!"
    
    print("[OK] Seed funktioniert - gleiche Seeds = gleiche Startbedingungen!")
    print(f"     Beide Environments: Haushalt 0 = {cash_env1[0]:.2f} EUR")
    print(f"     Beide Environments: Firma 0 = {capital_env1[0]:.2f} EUR\n")


def test_basic_simulation():
    """Test: Basis-Simulation laeuft"""
    print("[TEST 3] Basis-Simulation\n")
    
    # Mit festem Seed fuer konsistente Tests
    env = SimpleEconomyEnv(seed=123)
    obs, info = env.reset()
    
    print(f"Start: {info['total_household_cash']:.0f} EUR (Haushalte), {info['total_firm_capital']:.0f} EUR (Firmen)")
    
    # 50 Steps simulieren
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"  Tag {info['day']}: "
                  f"Pleite H={info['bankrupt_households']}, F={info['bankrupt_firms']}, "
                  f"Reward={reward:.2f}")
        
        if terminated:
            print(f"\n[WARN] Episode nach {i+1} Steps beendet")
            break
    
    print("\n[OK] Simulation erfolgreich!\n")


if __name__ == "__main__":
    print("="*60)
    print("Simple Economy Environment - Tests")
    print("="*60 + "\n")
    
    try:
        test_fixed_initial_conditions()
        test_seed_reproducibility()
        test_basic_simulation()
        
        print("="*60)
        print("[SUCCESS] Alle Tests bestanden!")
        print("="*60)
    except Exception as e:
        print(f"[FAILED] Test fehlgeschlagen: {e}")
        raise
