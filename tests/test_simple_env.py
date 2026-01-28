"""Test-Script fuer SimpleEconomyEnv (Multi-Agent)

Prueft:
1. Multi-Agent Struktur (jeder Agent eigener Observation/Action)
2. Verschiedene Seeds pro Episode (Variation)
3. Basis-Simulation mit Multi-Agent Actions
"""

import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.simple_economy_env import SimpleEconomyEnv


def test_multi_agent_structure():
    """Test: Multi-Agent Struktur funktioniert"""
    print("[TEST 1] Multi-Agent Struktur\n")
    
    env = SimpleEconomyEnv()
    obs, info = env.reset()
    
    # Pruefe: Observations fuer alle Agents
    expected_agents = env.num_households + env.num_firms
    assert len(obs) == expected_agents, f"Expected {expected_agents} agents, got {len(obs)}"
    
    # Pruefe: Haushalte haben 3-dim Observations
    for i in range(env.num_households):
        key = f'household_{i}'
        assert key in obs, f"Missing observation for {key}"
        assert obs[key].shape == (3,), f"Household obs should be 3-dim, got {obs[key].shape}"
    
    # Pruefe: Firmen haben 4-dim Observations
    for i in range(env.num_firms):
        key = f'firm_{i}'
        assert key in obs, f"Missing observation for {key}"
        assert obs[key].shape == (4,), f"Firm obs should be 4-dim, got {obs[key].shape}"
    
    print("[OK] Multi-Agent Struktur korrekt!")
    print(f"     Total Agents: {expected_agents}")
    print(f"     Haushalte: {env.num_households}")
    print(f"     Firmen: {env.num_firms}")
    print(f"     Household obs shape: (3,) [cash, avg_price, employed]")
    print(f"     Firm obs shape: (4,) [capital, inventory, employees, demand]\n")


def test_random_seeds_per_episode():
    """Test: Jede Episode hat verschiedene Startbedingungen"""
    print("[TEST 2] Verschiedene Seeds pro Episode\n")
    
    env = SimpleEconomyEnv()
    
    # Episode 1
    obs1, _ = env.reset()
    cash1 = obs1['household_0'][0]  # Cash von Haushalt 0
    
    # Episode 2
    obs2, _ = env.reset()
    cash2 = obs2['household_0'][0]
    
    # Episode 3
    obs3, _ = env.reset()
    cash3 = obs3['household_0'][0]
    
    print(f"Episode 1 - Haushalt 0: {cash1:.2f} EUR")
    print(f"Episode 2 - Haushalt 0: {cash2:.2f} EUR")
    print(f"Episode 3 - Haushalt 0: {cash3:.2f} EUR")
    
    # Pruefe: Sollten UNTERSCHIEDLICH sein (mit hoher Wahrscheinlichkeit)
    # Wenn alle 3 gleich sind, ist was falsch
    all_same = (cash1 == cash2 == cash3)
    
    if all_same:
        print("\n[WARN] Alle 3 Episoden haben gleiche Werte - unwahrscheinlich bei Random!")
    else:
        print("\n[OK] Episoden haben verschiedene Startwerte (Random Seeds funktionieren!)\n")


def test_multi_agent_actions():
    """Test: Multi-Agent Actions funktionieren"""
    print("[TEST 3] Multi-Agent Actions\n")
    
    env = SimpleEconomyEnv()
    obs, info = env.reset()
    
    print(f"Start: {info['total_household_cash']:.0f} EUR (Haushalte), {info['total_firm_capital']:.0f} EUR (Firmen)")
    
    # 10 Steps mit Dummy-Actions
    for i in range(10):
        actions = env._create_dummy_actions()
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if i % 5 == 0:
            # Zeige ein paar Rewards
            h0_reward = rewards['household_0']
            f0_reward = rewards['firm_0']
            print(f"  Tag {info['day']}: "
                  f"H0_reward={h0_reward:.2f}, F0_reward={f0_reward:.2f}, "
                  f"Pleite: H={info['bankrupt_households']}, F={info['bankrupt_firms']}")
        
        if terminated:
            print(f"\n[WARN] Episode nach {i+1} Steps beendet")
            break
    
    print("\n[OK] Multi-Agent Actions funktionieren!\n")


def test_action_spaces():
    """Test: Action/Observation Spaces sind korrekt definiert"""
    print("[TEST 4] Action/Observation Spaces\n")
    
    env = SimpleEconomyEnv()
    
    # Household Spaces
    print("Household Spaces:")
    print(f"  Observation: {env.household_observation_space}")
    print(f"  Action: {env.household_action_space}")
    
    # Firm Spaces
    print("\nFirm Spaces:")
    print(f"  Observation: {env.firm_observation_space}")
    print(f"  Action: {env.firm_action_space}")
    
    # Pruefe Dimensionen
    assert env.household_observation_space.shape == (3,), "Household obs should be 3-dim"
    assert env.household_action_space.shape == (1,), "Household action should be 1-dim"
    assert env.firm_observation_space.shape == (4,), "Firm obs should be 4-dim"
    assert env.firm_action_space.shape == (3,), "Firm action should be 3-dim"
    
    print("\n[OK] Spaces korrekt definiert!\n")


if __name__ == "__main__":
    print("="*60)
    print("Multi-Agent Economy Environment - Tests")
    print("="*60 + "\n")
    
    try:
        test_multi_agent_structure()
        test_random_seeds_per_episode()
        test_multi_agent_actions()
        test_action_spaces()
        
        print("="*60)
        print("[SUCCESS] Alle Tests bestanden!")
        print("="*60)
    except Exception as e:
        print(f"\n[FAILED] Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        raise
