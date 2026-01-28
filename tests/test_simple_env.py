"""Test-Script für SimpleEconomyEnv

Führt einen kurzen Test durch:
1. Environment initialisieren
2. Prüft ob initiale Bedingungen fix bleiben
3. Simuliert ein paar Steps
"""

import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.simple_economy_env import SimpleEconomyEnv


def test_fixed_initial_conditions():
    """Test: Initiale Bedingungen bleiben über Episoden fix"""
    print("\ud83e\uddea Test: Fixe Startbedingungen\n")
    
    env = SimpleEconomyEnv()
    
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
    
    print("\u2705 Startbedingungen bleiben über Episoden fix!")
    print(f"   Haushalt 0: Immer {household_cash_ep1[0]:.2f}€")
    print(f"   Firma 0: Immer {firm_capital_ep1[0]:.2f}€\n")


def test_basic_simulation():
    """Test: Basis-Simulation läuft"""
    print("\ud83e\uddea Test: Basis-Simulation\n")
    
    env = SimpleEconomyEnv()
    obs, info = env.reset()
    
    print(f"Start: {info['total_household_cash']:.0f}€ (Haushalte), {info['total_firm_capital']:.0f}€ (Firmen)")
    
    # 50 Steps simulieren
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"  Tag {info['day']}: "
                  f"Pleite H={info['bankrupt_households']}, F={info['bankrupt_firms']}, "
                  f"Reward={reward:.2f}")
        
        if terminated:
            print(f"\n\u26a0\ufe0f Episode nach {i+1} Steps beendet")
            break
    
    print("\n\u2705 Simulation erfolgreich!\n")


if __name__ == "__main__":
    print("="*60)
    print("Simple Economy Environment - Tests")
    print("="*60 + "\n")
    
    try:
        test_fixed_initial_conditions()
        test_basic_simulation()
        
        print("\u2705 Alle Tests bestanden!")
    except Exception as e:
        print(f"\u274c Test fehlgeschlagen: {e}")
        raise
