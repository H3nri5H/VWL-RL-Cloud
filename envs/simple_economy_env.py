"""Simple Multi-Agent Economy Environment

NUR Haushalte und Unternehmen - KEIN Staat (erstmal)
Sehr basic - kann spaeter erweitert werden
"""

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from pathlib import Path


class SimpleEconomyEnv(gym.Env):
    """Simples Wirtschafts-Environment mit Haushalten und Firmen"""
    
    metadata = {'render_modes': []}
    
    def __init__(self, config_path='configs/agent_config.yaml', seed=None):
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
        
        # Seed fuer Reproduzierbarkeit
        self.init_seed = seed
        
        # Zeitvariablen
        self.current_step = 0
        self.current_day = 0
        
        # Initiale Bedingungen werden EINMAL beim Init gesetzt
        self._initialize_initial_conditions()
        
        # Action/Observation Spaces (erstmal placeholder - wird spaeter definiert)
        # TODO: Richtig definieren wenn Aktionen klar sind
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # Agents
        self.households = []
        self.firms = []
    
    def _initialize_initial_conditions(self):
        """Erstelle fixe Startbedingungen (NUR EINMAL beim Training-Start)
        
        Wenn seed gesetzt ist, sind die Startbedingungen reproduzierbar.
        """
        
        # Seed setzen fuer reproduzierbare Startbedingungen
        if self.init_seed is not None:
            np.random.seed(self.init_seed)
        
        # Haushalte: Jeder kriegt zufaelligen Wert aus Range
        h_config = self.config['households']
        self.initial_household_cash = [
            np.random.uniform(
                h_config['initial_cash']['min'],
                h_config['initial_cash']['max']
            )
            for _ in range(self.num_households)
        ]
        
        # Firmen: Jede kriegt zufaellige Werte aus Range
        f_config = self.config['firms']
        self.initial_firm_capital = [
            np.random.uniform(
                f_config['initial_capital']['min'],
                f_config['initial_capital']['max']
            )
            for _ in range(self.num_firms)
        ]
        self.initial_firm_employees = [
            np.random.randint(
                f_config['initial_employees']['min'],
                f_config['initial_employees']['max'] + 1
            )
            for _ in range(self.num_firms)
        ]
        
        # Seed zuruecksetzen damit nachfolgende Operations nicht deterministisch sind
        if self.init_seed is not None:
            np.random.seed(None)
        
        seed_info = f" (seed={self.init_seed})" if self.init_seed is not None else " (random)"
        print(f"[OK] Initiale Bedingungen erstellt{seed_info}:")
        print(f"     Haushalte: {self.num_households} mit Cash {min(self.initial_household_cash):.0f} EUR - {max(self.initial_household_cash):.0f} EUR")
        print(f"     Firmen: {self.num_firms} mit Kapital {min(self.initial_firm_capital):.0f} EUR - {max(self.initial_firm_capital):.0f} EUR")
    
    def reset(self, seed=None, options=None):
        """Zurueck zu initialen Bedingungen (werden NICHT neu gewuerfelt!)"""
        super().reset(seed=seed)
        
        # Zeit zuruecksetzen
        self.current_step = 0
        self.current_day = 0
        
        # Haushalte mit FIXEN Startbedingungen
        self.households = [
            {
                'id': i,
                'cash': self.initial_household_cash[i],  # IMMER GLEICH!
                'employed': True,
                'bankrupt': False
            }
            for i in range(self.num_households)
        ]
        
        # Firmen mit FIXEN Startbedingungen
        self.firms = [
            {
                'id': i,
                'capital': self.initial_firm_capital[i],      # IMMER GLEICH!
                'employees': self.initial_firm_employees[i],  # IMMER GLEICH!
                'inventory': 100.0,
                'bankrupt': False
            }
            for i in range(self.num_firms)
        ]
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Simuliere einen Betriebstag"""
        
        # === PLACEHOLDER - Hier kommt spaeter die Logik ===
        # TODO: 
        # 1. Firmen produzieren
        # 2. Haushalte konsumieren
        # 3. Markt-Clearing
        # 4. Rewards berechnen
        
        # Fuer jetzt: Simple Dummy-Logik
        for household in self.households:
            if not household['bankrupt']:
                # Haushalte konsumieren etwas
                consumption = household['cash'] * 0.1
                household['cash'] -= consumption
                
                # Pleite-Check
                if household['cash'] < 0:
                    household['bankrupt'] = True
        
        for firm in self.firms:
            if not firm['bankrupt']:
                # Firmen zahlen Loehne
                wages = firm['employees'] * 50
                firm['capital'] -= wages
                
                # Pleite-Check
                if firm['capital'] < 0:
                    firm['bankrupt'] = True
        
        # Zeit vorwaerts
        self.current_step += 1
        self.current_day += 1
        
        # Episode Ende?
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        # Reward (placeholder)
        reward = 1.0 if not any(h['bankrupt'] for h in self.households) else -1.0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Aktueller State (placeholder)"""
        # TODO: Richtige Observations definieren
        return np.zeros(10, dtype=np.float32)
    
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
            'total_firm_capital': sum(f['capital'] for f in self.firms if not f['bankrupt'])
        }


if __name__ == "__main__":
    # Test
    print("[TEST] Testing SimpleEconomyEnv...\n")
    
    # Mit Seed - reproduzierbar
    print("=== Test mit Seed ===\n")
    env1 = SimpleEconomyEnv(seed=42)
    obs1, info1 = env1.reset()
    print(f"Haushalt 0: {env1.households[0]['cash']:.2f} EUR\n")
    
    # Gleiches Seed - sollte gleiche Werte haben!
    env2 = SimpleEconomyEnv(seed=42)
    obs2, info2 = env2.reset()
    print(f"Haushalt 0 (Env2): {env2.households[0]['cash']:.2f} EUR")
    
    if env1.households[0]['cash'] == env2.households[0]['cash']:
        print("[OK] Seeds funktionieren - Werte sind identisch!\n")
    else:
        print("[ERROR] Seeds funktionieren nicht!\n")
    
    # Ohne Seed - random
    print("=== Test ohne Seed (random) ===\n")
    env3 = SimpleEconomyEnv()
    obs3, info3 = env3.reset()
    print(f"Haushalt 0 (Env3): {env3.households[0]['cash']:.2f} EUR\n")
    
    # Ein paar Steps
    print("=== Steps Test ===\n")
    for i in range(5):
        action = env1.action_space.sample()
        obs, reward, terminated, truncated, info = env1.step(action)
        print(f"Step {i+1}: Reward={reward:.2f}")
        
        if terminated:
            break
    
    print("\n[OK] Test completed!")
