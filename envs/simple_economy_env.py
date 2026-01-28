"""Simple Multi-Agent Economy Environment

Jeder Agent (Haushalt/Firma) hat eigenen Observation/Action Space.
Seeds werden pro Episode neu gezogen fuer Variation.
"""

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from pathlib import Path


class SimpleEconomyEnv(gym.Env):
    """Multi-Agent Wirtschafts-Environment
    
    Jeder Haushalt und jede Firma ist ein eigener Agent.
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, config_path='configs/agent_config.yaml'):
        super().__init__()
        
        # Config laden
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Parameter
        self.num_households = self.config['households']['count']
        self.num_firms = self.config['firms']['count']
        self.days_per_year = self.config['simulation']['days_per_year']
        self.max_years = self.config['simulation']['max_years']
        self.max_steps = self.days_per_year * self.max_years
        
        # Zeitvariablen
        self.current_step = 0
        self.current_day = 0
        
        # Multi-Agent: Jeder Agent hat eigenen Space
        self._setup_action_observation_spaces()
        
        # Agents
        self.households = []
        self.firms = []
        
        print(f"[OK] Environment initialisiert:")
        print(f"     Haushalte: {self.num_households} (je eigener Agent)")
        print(f"     Firmen: {self.num_firms} (je eigener Agent)")
        print(f"     Total Agents: {self.num_households + self.num_firms}")
    
    def _setup_action_observation_spaces(self):
        """Definiere Action/Observation Spaces pro Agent-Typ"""
        
        # === HOUSEHOLD SPACES ===
        # Observation: [eigenes_cash, durchschnittspreis, employed (0/1)]
        self.household_observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([100000.0, 50.0, 1.0]),
            dtype=np.float32
        )
        
        # Action: [konsumquote] (0-100%)
        self.household_action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # === FIRM SPACES ===
        # Observation: [kapital, inventar, mitarbeiter, nachfrage]
        self.firm_observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1000000.0, 1000.0, 50.0, 1000.0]),
            dtype=np.float32
        )
        
        # Action: [produktion_menge, preis, mitarbeiter_change]
        # produktion: 0-200 Einheiten
        # preis: 5-15 EUR
        # mitarbeiter_change: -2 bis +2
        self.firm_action_space = spaces.Box(
            low=np.array([0.0, 5.0, -2.0]),
            high=np.array([200.0, 15.0, 2.0]),
            dtype=np.float32
        )
        
        # Placeholder fuer Gym (wird spaeter mit RLlib ueberschrieben)
        self.observation_space = self.household_observation_space
        self.action_space = self.household_action_space
    
    def reset(self, seed=None, options=None):
        """Neue Episode mit NEUEM Seed (Variation!)"""
        super().reset(seed=seed)
        
        # Zeit zuruecksetzen
        self.current_step = 0
        self.current_day = 0
        
        # NEUER Seed pro Episode (fuer Variation)
        episode_seed = seed if seed is not None else np.random.randint(0, 1000000)
        np.random.seed(episode_seed)
        
        # Config laden
        h_config = self.config['households']
        f_config = self.config['firms']
        
        # Haushalte mit NEUEN zufaelligen Startwerten
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
        
        # Firmen mit NEUEN zufaelligen Startwerten
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
                'price': 10.0,  # Startpreis
                'bankrupt': False,
                'production_history': []
            }
            for i in range(self.num_firms)
        ]
        
        # Seed zuruecksetzen
        np.random.seed(None)
        
        print(f"\n[RESET] Episode Start (seed={episode_seed}):")
        print(f"        Haushalte: {min(h['cash'] for h in self.households):.0f} - {max(h['cash'] for h in self.households):.0f} EUR")
        print(f"        Firmen: {min(f['capital'] for f in self.firms):.0f} - {max(f['capital'] for f in self.firms):.0f} EUR")
        
        obs = self._get_multi_agent_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, actions):
        """Simuliere einen Tag mit Multi-Agent Actions
        
        Args:
            actions: Dict mit allen Agent-Actions
                     {'household_0': [0.1], 'household_1': [0.2], ...}
                     oder einzelne Action (fuer Testing)
        """
        
        # Fallback fuer Testing (wenn nur eine Action kommt)
        if not isinstance(actions, dict):
            actions = self._create_dummy_actions()
        
        # === 1. HAUSHALTE KONSUMIEREN ===
        for i, household in enumerate(self.households):
            if household['bankrupt']:
                continue
            
            action_key = f'household_{i}'
            if action_key in actions:
                konsumquote = float(actions[action_key][0])
            else:
                konsumquote = 0.1  # Fallback
            
            # Konsumieren
            consumption = household['cash'] * konsumquote
            household['cash'] -= consumption
            household['consumption_history'].append(consumption)
            
            # Pleite-Check
            if household['cash'] < 0:
                household['bankrupt'] = True
        
        # === 2. FIRMEN AGIEREN ===
        for i, firm in enumerate(self.firms):
            if firm['bankrupt']:
                continue
            
            action_key = f'firm_{i}'
            if action_key in actions:
                production = float(actions[action_key][0])
                price = float(actions[action_key][1])
                employee_change = int(actions[action_key][2])
            else:
                production = 50.0  # Fallback
                price = 10.0
                employee_change = 0
            
            # Produktion
            production_cost = production * 2.0  # 2 EUR pro Einheit
            firm['capital'] -= production_cost
            firm['inventory'] += production
            firm['production_history'].append(production)
            
            # Mitarbeiter anpassen
            firm['employees'] = max(0, firm['employees'] + employee_change)
            
            # Loehne zahlen
            wages = firm['employees'] * 50.0
            firm['capital'] -= wages
            
            # Preis setzen
            firm['price'] = price
            
            # Pleite-Check
            if firm['capital'] < 0:
                firm['bankrupt'] = True
        
        # === 3. MARKT (placeholder) ===
        # TODO: Markt-Clearing zwischen Haushalten und Firmen
        
        # Zeit vorwaerts
        self.current_step += 1
        self.current_day += 1
        
        # Episode Ende?
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        # Multi-Agent Rewards
        rewards = self._calculate_rewards()
        
        obs = self._get_multi_agent_observation()
        info = self._get_info()
        
        return obs, rewards, terminated, truncated, info
    
    def _create_dummy_actions(self):
        """Erstelle Dummy-Actions fuer Testing"""
        actions = {}
        
        for i in range(self.num_households):
            actions[f'household_{i}'] = np.array([0.1])  # 10% konsumieren
        
        for i in range(self.num_firms):
            actions[f'firm_{i}'] = np.array([50.0, 10.0, 0.0])  # 50 Einheiten, 10 EUR, 0 Mitarbeiter-Change
        
        return actions
    
    def _get_multi_agent_observation(self):
        """Observations fuer alle Agents"""
        obs = {}
        
        # Markt-Info (alle sehen das)
        avg_price = np.mean([f['price'] for f in self.firms if not f['bankrupt']])
        total_demand = sum(h['consumption_history'][-1] if h['consumption_history'] else 0 
                          for h in self.households if not h['bankrupt'])
        
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
        
        # Haushalte: Nutzen aus Konsum
        for i, household in enumerate(self.households):
            if household['bankrupt']:
                rewards[f'household_{i}'] = -10.0
            else:
                # Reward = Konsum + kleine Belohnung fuers Ueberleben
                consumption = household['consumption_history'][-1] if household['consumption_history'] else 0
                rewards[f'household_{i}'] = consumption * 0.1 + 1.0
        
        # Firmen: Profit
        for i, firm in enumerate(self.firms):
            if firm['bankrupt']:
                rewards[f'firm_{i}'] = -10.0
            else:
                # Reward = Capital-Change (vereinfacht)
                rewards[f'firm_{i}'] = firm['capital'] / 100000.0  # Skaliert
        
        return rewards
    
    def _get_info(self):
        """Zusaetzliche Infos"""
        num_bankrupt_households = sum(1 for h in self.households if h['bankrupt'])
        num_bankrupt_firms = sum(1 for f in self.firms if f['bankrupt'])
        
        return {
            'step': self.current_step,
            'day': self.current_day,
            'bankrupt_households': num_bankrupt_households,
            'bankrupt_firms': num_bankrupt_firms,
            'total_household_cash': sum(h['cash'] for h in self.households if not h['bankrupt']),
            'total_firm_capital': sum(f['capital'] for f in self.firms if not f['bankrupt']),
            'num_agents': self.num_households + self.num_firms
        }


if __name__ == "__main__":
    print("[TEST] Testing Multi-Agent Environment...\n")
    
    env = SimpleEconomyEnv()
    
    # Episode 1
    print("\n=== Episode 1 ===")
    obs, info = env.reset()
    print(f"Agents: {len(obs)}")
    print(f"Household 0 obs: {obs['household_0']}")
    print(f"Firm 0 obs: {obs['firm_0']}")
    
    # Ein paar Steps
    for i in range(3):
        actions = env._create_dummy_actions()
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"\nStep {i+1}:")
        print(f"  Household 0 reward: {rewards['household_0']:.2f}")
        print(f"  Firm 0 reward: {rewards['firm_0']:.2f}")
    
    # Episode 2 - ANDERER Seed!
    print("\n=== Episode 2 ===")
    obs, info = env.reset()
    print(f"Household 0 obs: {obs['household_0']}")
    
    print("\n[OK] Multi-Agent Test completed!")
