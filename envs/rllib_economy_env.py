"""RLlib-compatible Multi-Agent Economy Environment

Wrapper um SimpleEconomyEnv fuer RLlib Training.
"""

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import yaml
from gymnasium import spaces
from pathlib import Path


class RLlibEconomyEnv(MultiAgentEnv):
    """RLlib Multi-Agent Environment fuer Wirtschafts-Simulation
    
    Kompatibel mit Ray RLlib fuer Multi-Agent RL Training.
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Config laden
        config_path = config.get('config_path', 'configs/agent_config.yaml') if config else 'configs/agent_config.yaml'
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            self.env_config = yaml.safe_load(f)
        
        # Parameter
        self.num_households = self.env_config['households']['count']
        self.num_firms = self.env_config['firms']['count']
        self.days_per_year = self.env_config['simulation']['days_per_year']
        self.max_years = self.env_config['simulation']['max_years']
        self.max_steps = self.days_per_year * self.max_years
        
        # Agent IDs
        self._agent_ids = set()
        for i in range(self.num_households):
            self._agent_ids.add(f'household_{i}')
        for i in range(self.num_firms):
            self._agent_ids.add(f'firm_{i}')
        
        # Zeitvariablen
        self.current_step = 0
        self.current_day = 0
        
        # Action/Observation Spaces
        self._setup_spaces()
        
        # Agents
        self.households = []
        self.firms = []
        
        print(f"[RLlib] Environment initialisiert:")
        print(f"        Haushalte: {self.num_households}")
        print(f"        Firmen: {self.num_firms}")
        print(f"        Total Agents: {len(self._agent_ids)}")
    
    def _setup_spaces(self):
        """Action/Observation Spaces definieren"""
        
        # Household Spaces
        self.household_observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([100000.0, 50.0, 1.0]),
            dtype=np.float32
        )
        self.household_action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # Firm Spaces
        self.firm_observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1000000.0, 1000.0, 50.0, 1000.0]),
            dtype=np.float32
        )
        self.firm_action_space = spaces.Box(
            low=np.array([0.0, 5.0, -2.0]),
            high=np.array([200.0, 15.0, 2.0]),
            dtype=np.float32
        )
    
    def reset(self, *, seed=None, options=None):
        """Reset environment fuer neue Episode"""
        
        self.current_step = 0
        self.current_day = 0
        
        # Random Seed pro Episode
        episode_seed = seed if seed is not None else np.random.randint(0, 1000000)
        np.random.seed(episode_seed)
        
        h_config = self.env_config['households']
        f_config = self.env_config['firms']
        
        # Haushalte initialisieren
        self.households = [
            {
                'id': i,
                'cash': np.random.uniform(
                    h_config['initial_cash']['min'],
                    h_config['initial_cash']['max']
                ),
                'employed': True,
                'bankrupt': False,
                'consumption_history': []
            }
            for i in range(self.num_households)
        ]
        
        # Firmen initialisieren
        self.firms = [
            {
                'id': i,
                'capital': np.random.uniform(
                    f_config['initial_capital']['min'],
                    f_config['initial_capital']['max']
                ),
                'employees': np.random.randint(
                    f_config['initial_employees']['min'],
                    f_config['initial_employees']['max'] + 1
                ),
                'inventory': 100.0,
                'price': 10.0,
                'bankrupt': False,
                'production_history': []
            }
            for i in range(self.num_firms)
        ]
        
        np.random.seed(None)
        
        obs = self._get_observations()
        infos = {agent_id: {} for agent_id in self._agent_ids}
        
        return obs, infos
    
    def step(self, action_dict):
        """Environment Step mit Multi-Agent Actions
        
        Args:
            action_dict: {agent_id: action, ...}
        
        Returns:
            obs: {agent_id: observation, ...}
            rewards: {agent_id: reward, ...}
            terminateds: {agent_id: bool, ...}
            truncateds: {agent_id: bool, ...}
            infos: {agent_id: info_dict, ...}
        """
        
        # Haushalte konsumieren
        for i, household in enumerate(self.households):
            if household['bankrupt']:
                continue
            
            agent_id = f'household_{i}'
            if agent_id in action_dict:
                konsumquote = float(action_dict[agent_id][0])
            else:
                konsumquote = 0.1
            
            consumption = household['cash'] * konsumquote
            household['cash'] -= consumption
            household['consumption_history'].append(consumption)
            
            if household['cash'] < 0:
                household['bankrupt'] = True
        
        # Firmen agieren
        for i, firm in enumerate(self.firms):
            if firm['bankrupt']:
                continue
            
            agent_id = f'firm_{i}'
            if agent_id in action_dict:
                production = float(action_dict[agent_id][0])
                price = float(action_dict[agent_id][1])
                employee_change = int(action_dict[agent_id][2])
            else:
                production = 50.0
                price = 10.0
                employee_change = 0
            
            # Produktion
            production_cost = production * 2.0
            firm['capital'] -= production_cost
            firm['inventory'] += production
            firm['production_history'].append(production)
            
            # Mitarbeiter
            firm['employees'] = max(0, firm['employees'] + employee_change)
            
            # Loehne
            wages = firm['employees'] * 50.0
            firm['capital'] -= wages
            
            # Preis
            firm['price'] = price
            
            if firm['capital'] < 0:
                firm['bankrupt'] = True
        
        # Zeit
        self.current_step += 1
        self.current_day += 1
        
        # Episode Ende?
        done = (self.current_step >= self.max_steps)
        
        # Observations, Rewards, etc.
        obs = self._get_observations()
        rewards = self._calculate_rewards()
        
        # RLlib Format: terminateds und truncateds per agent
        terminateds = {agent_id: done for agent_id in self._agent_ids}
        terminateds['__all__'] = done
        
        truncateds = {agent_id: False for agent_id in self._agent_ids}
        truncateds['__all__'] = False
        
        infos = {agent_id: self._get_info() for agent_id in self._agent_ids}
        
        return obs, rewards, terminateds, truncateds, infos
    
    def _get_observations(self):
        """Observations fuer alle Agents"""
        obs = {}
        
        # Markt-Info
        avg_price = np.mean([f['price'] for f in self.firms if not f['bankrupt']]) if any(not f['bankrupt'] for f in self.firms) else 10.0
        total_demand = sum(h['consumption_history'][-1] if h['consumption_history'] else 0 for h in self.households if not h['bankrupt'])
        
        # Haushalte
        for i, household in enumerate(self.households):
            obs[f'household_{i}'] = np.array([
                household['cash'],
                avg_price,
                1.0 if household['employed'] else 0.0
            ], dtype=np.float32)
        
        # Firmen
        for i, firm in enumerate(self.firms):
            obs[f'firm_{i}'] = np.array([
                firm['capital'],
                firm['inventory'],
                float(firm['employees']),
                total_demand
            ], dtype=np.float32)
        
        return obs
    
    def _calculate_rewards(self):
        """Rewards fuer alle Agents"""
        rewards = {}
        
        # Haushalte
        for i, household in enumerate(self.households):
            if household['bankrupt']:
                rewards[f'household_{i}'] = -10.0
            else:
                consumption = household['consumption_history'][-1] if household['consumption_history'] else 0
                rewards[f'household_{i}'] = consumption * 0.1 + 1.0
        
        # Firmen
        for i, firm in enumerate(self.firms):
            if firm['bankrupt']:
                rewards[f'firm_{i}'] = -10.0
            else:
                rewards[f'firm_{i}'] = firm['capital'] / 100000.0
        
        return rewards
    
    def _get_info(self):
        """Info Dictionary"""
        return {
            'step': self.current_step,
            'day': self.current_day,
            'bankrupt_households': sum(1 for h in self.households if h['bankrupt']),
            'bankrupt_firms': sum(1 for f in self.firms if f['bankrupt'])
        }
    
    def get_agent_ids(self):
        """RLlib Interface: Return agent IDs"""
        return self._agent_ids
    
    def observation_space_sample(self, agent_ids=None):
        """RLlib Interface: Sample observation space"""
        if agent_ids is None:
            agent_ids = self._agent_ids
        
        obs = {}
        for agent_id in agent_ids:
            if 'household' in agent_id:
                obs[agent_id] = self.household_observation_space.sample()
            else:
                obs[agent_id] = self.firm_observation_space.sample()
        return obs
    
    def action_space_sample(self, agent_ids=None):
        """RLlib Interface: Sample action space"""
        if agent_ids is None:
            agent_ids = self._agent_ids
        
        actions = {}
        for agent_id in agent_ids:
            if 'household' in agent_id:
                actions[agent_id] = self.household_action_space.sample()
            else:
                actions[agent_id] = self.firm_action_space.sample()
        return actions
    
    def observation_space_contains(self, x):
        """RLlib Interface: Check if observation is valid"""
        for agent_id, obs in x.items():
            if 'household' in agent_id:
                if not self.household_observation_space.contains(obs):
                    return False
            else:
                if not self.firm_observation_space.contains(obs):
                    return False
        return True
    
    def action_space_contains(self, x):
        """RLlib Interface: Check if action is valid"""
        for agent_id, action in x.items():
            if 'household' in agent_id:
                if not self.household_action_space.contains(action):
                    return False
            else:
                if not self.firm_action_space.contains(action):
                    return False
        return True


if __name__ == "__main__":
    print("[TEST] RLlib Environment Test\n")
    
    env = RLlibEconomyEnv()
    
    obs, infos = env.reset()
    print(f"Agents: {len(obs)}")
    print(f"Household 0: {obs['household_0']}")
    print(f"Firm 0: {obs['firm_0']}\n")
    
    # Sample actions
    actions = env.action_space_sample()
    
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    print(f"Rewards:")
    print(f"  Household 0: {rewards['household_0']:.2f}")
    print(f"  Firm 0: {rewards['firm_0']:.2f}")
    
    print("\n[OK] RLlib Environment funktioniert!")
