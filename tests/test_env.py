"""Tests für EconomyEnv (ohne pytest)"""
import sys
from pathlib import Path

# PYTHONPATH Fix
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from envs.economy_env import EconomyEnv


def test_env_creation():
    """Test: Environment kann erstellt werden"""
    env = EconomyEnv()
    assert env is not None
    assert env.num_firms == 10
    assert env.num_households == 50


def test_env_reset():
    """Test: Reset funktioniert"""
    env = EconomyEnv()
    obs, info = env.reset()
    
    # Check observation shape
    assert obs.shape == (5,), f"Expected shape (5,), got {obs.shape}"
    
    # Check observation values
    assert obs[0] > 0, "BIP sollte positiv sein"
    assert -0.5 <= obs[1] <= 0.5, "Inflation außerhalb Range"
    assert 0 <= obs[2] <= 1.0, "Arbeitslosigkeit außerhalb Range"
    
    # Check info dict
    assert 'bip' in info
    assert 'inflation' in info
    assert 'unemployment' in info


def test_env_step():
    """Test: Step funktioniert"""
    env = EconomyEnv()
    obs, info = env.reset()
    
    # Valide Aktion: [Steuersatz, Staatsausgaben, Zinssatz]
    action = np.array([0.3, 500.0, 0.05], dtype=np.float32)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check types
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Check reward bounds
    assert -20.0 <= reward <= 10.0, f"Reward {reward} außerhalb [-20, 10]"


def test_env_episode():
    """Test: Ganze Episode läuft durch"""
    env = EconomyEnv()
    obs, info = env.reset()
    
    total_reward = 0
    steps = 0
    
    for i in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    assert steps > 0, "Episode sollte mindestens 1 Step haben"
    print(f"✅ Episode: {steps} steps, total reward: {total_reward:.2f}")


def test_env_action_clipping():
    """Test: Actions werden korrekt geclipped"""
    env = EconomyEnv()
    obs, info = env.reset()
    
    # Extreme Actions (sollten geclipped werden)
    action = np.array([1.0, 5000.0, 0.5], dtype=np.float32)  # Alle zu hoch
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Environment sollte nicht crashen
    assert obs is not None
    assert not np.isnan(reward)


def test_env_spaces():
    """Test: Action & Observation Spaces sind korrekt"""
    env = EconomyEnv()
    
    # Observation Space
    assert env.observation_space.shape == (5,)
    
    # Action Space
    assert env.action_space.shape == (3,)
    assert np.all(env.action_space.low >= 0)
    assert np.all(env.action_space.high > 0)


if __name__ == "__main__":
    print("\n🧪 Running EconomyEnv Tests...\n")
    
    try:
        test_env_creation()
        print("✅ test_env_creation")
        
        test_env_reset()
        print("✅ test_env_reset")
        
        test_env_step()
        print("✅ test_env_step")
        
        test_env_episode()
        print("✅ test_env_episode")
        
        test_env_action_clipping()
        print("✅ test_env_action_clipping")
        
        test_env_spaces()
        print("✅ test_env_spaces")
        
        print("\n🎉 All tests passed!")
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
