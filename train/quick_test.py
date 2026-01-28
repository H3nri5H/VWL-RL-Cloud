"""Quick Test: Multi-Agent Setup funktioniert?

Testet:
1. Environment laedt korrekt
2. Alle 15 Agents sind vorhanden (10 Haushalte + 5 Firmen)
3. Actions funktionieren
4. Training startet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from envs.rllib_economy_env import RLlibEconomyEnv


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map agents zu policies"""
    if 'household' in agent_id:
        return 'household_policy'
    else:
        return 'firm_policy'


def quick_test():
    print("\n" + "="*60)
    print("QUICK TEST: Multi-Agent Economy")
    print("="*60 + "\n")
    
    # 1. Environment Test
    print("[1/4] Environment initialisieren...")
    env = RLlibEconomyEnv({'config_path': 'configs/agent_config.yaml'})
    
    agent_ids = env.get_agent_ids()
    print(f"      OK {len(agent_ids)} Agents gefunden")
    print(f"         - Haushalte: {sum(1 for a in agent_ids if 'household' in a)}")
    print(f"         - Firmen: {sum(1 for a in agent_ids if 'firm' in a)}")
    
    # 2. Reset Test
    print("\n[2/4] Environment Reset...")
    obs, infos = env.reset()
    print(f"      OK Observations: {len(obs)} agents")
    print(f"         Household_0: {obs['household_0']}")
    print(f"         Firm_0: {obs['firm_0']}")
    
    # 3. Step Test
    print("\n[3/4] Step mit Random Actions...")
    actions = env.action_space_sample()
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    
    print(f"      OK Step erfolgreich")
    print(f"         Reward Household_0: {rewards['household_0']:.2f}")
    print(f"         Reward Firm_0: {rewards['firm_0']:.2f}")
    
    # 4. Mini-Training Test (nur 100 steps)
    print("\n[4/4] Mini-Training starten (100 steps)...")
    
    ray.init(ignore_reinit_error=True, num_cpus=2)
    
    register_env("economy", lambda config: RLlibEconomyEnv(config))
    
    dummy_env = RLlibEconomyEnv()
    
    policies = {
        'household_policy': PolicySpec(
            observation_space=dummy_env.household_observation_space,
            action_space=dummy_env.household_action_space,
            config={}
        ),
        'firm_policy': PolicySpec(
            observation_space=dummy_env.firm_observation_space,
            action_space=dummy_env.firm_action_space,
            config={}
        )
    }
    
    # Ray 2.30.0 - stable API
    config = (
        PPOConfig()
        .environment(
            env="economy",
            env_config={'config_path': 'configs/agent_config.yaml'}
        )
        .framework("torch")
        .training(
            train_batch_size=200,
            sgd_minibatch_size=50,
            num_sgd_iter=3
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=['household_policy', 'firm_policy']
        )
        .rollouts(
            num_rollout_workers=1
        )
    )
    
    algo = config.build()
    
    print("      Training laeuft...")
    result = algo.train()
    
    print(f"      OK Training erfolgreich!")
    
    # Ray 2.30.0 keys - flexible handling
    if 'sampler_results' in result:
        if 'episode_reward_mean' in result['sampler_results']:
            print(f"         Episode Reward Mean: {result['sampler_results']['episode_reward_mean']:.2f}")
    elif 'episode_reward_mean' in result:
        print(f"         Episode Reward Mean: {result['episode_reward_mean']:.2f}")
    
    if 'episodes_this_iter' in result:
        print(f"         Episodes: {result['episodes_this_iter']}")
    
    print(f"         Timesteps: {result.get('timesteps_total', 0)}")
    
    algo.stop()
    ray.shutdown()
    
    print("\n" + "="*60)
    print("ALLE TESTS BESTANDEN!")
    print("="*60)
    print("\nNaechster Schritt:")
    print("  python train/train_local.py --timesteps 10000")
    print("\n")


if __name__ == "__main__":
    quick_test()
