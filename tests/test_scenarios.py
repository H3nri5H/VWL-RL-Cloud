"""Szenario-Tests für verschiedene Wirtschaftslagen"""
import numpy as np
from envs.economy_env import EconomyEnv


def run_scenario(name, tax_rate, gov_spending, interest_rate, steps=50):
    """Führe Szenario aus mit festen Parametern"""
    print(f"\n📊 Szenario: {name}")
    print(f"   Tax: {tax_rate:.1%}, Spending: {gov_spending:.0f}, Interest: {interest_rate:.1%}")
    
    env = EconomyEnv()
    obs, info = env.reset()
    
    total_reward = 0
    bip_start = info['bip']
    
    for i in range(steps):
        action = np.array([tax_rate, gov_spending, interest_rate], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        if terminated:
            print(f"   ⚠️ Episode terminiert bei Step {i+1}")
            break
    
    bip_end = info['bip']
    bip_growth = ((bip_end / bip_start) - 1) * 100
    
    print(f"   BIP: {bip_start:.0f} → {bip_end:.0f} ({bip_growth:+.1f}%)")
    print(f"   Arbeitslosigkeit: {info['unemployment']:.1%}")
    print(f"   Inflation: {info['inflation']:.2%}")
    print(f"   Total Reward: {total_reward:.2f}")
    
    return total_reward


def test_scenarios():
    """Teste verschiedene Wirtschafts-Szenarien"""
    
    print("\n" + "="*60)
    print("🧪 VWL-RL Szenario-Tests")
    print("="*60)
    
    scenarios = [
        ("Normal", 0.30, 500, 0.05),
        ("Niedrige Steuern", 0.15, 500, 0.05),
        ("Hohe Steuern", 0.45, 500, 0.05),
        ("Hohe Staatsausgaben", 0.30, 800, 0.05),
        ("Niedrige Staatsausgaben", 0.30, 200, 0.05),
        ("Hohe Zinsen", 0.30, 500, 0.15),
        ("Niedrige Zinsen", 0.30, 500, 0.01),
        ("Austerität", 0.40, 200, 0.10),
        ("Expansion", 0.20, 800, 0.02),
    ]
    
    results = []
    
    for name, tax, spend, interest in scenarios:
        reward = run_scenario(name, tax, spend, interest)
        results.append((name, reward))
    
    # Ranking
    print("\n" + "="*60)
    print("🏆 Ranking nach Total Reward:")
    print("="*60)
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, reward) in enumerate(results, 1):
        print(f"{i}. {name:25s} | Reward: {reward:+8.2f}")
    
    print("\n✅ Szenario-Tests abgeschlossen!\n")


if __name__ == "__main__":
    test_scenarios()
