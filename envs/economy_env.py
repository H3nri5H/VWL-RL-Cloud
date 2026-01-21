import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EconomyEnv(gym.Env):
    """
    Volkswirtschafts-Simulation mit Multi-Agent RL
    
    Agents:
    - 10 Firmen (RL): Preis, Löhne, Mitarbeiteranzahl
    - 50 Haushalte (regelbasiert): Konsum basierend auf Einkommen
    - 1 Regierung (RL): Steuern, Staatsausgaben, Zinsen
    
    Observation: [BIP, Inflation, Arbeitslosigkeit, Kapital, ...]
    Action: [Steuersatz, Staatsausgaben, Zinssatz] für Regierung
    Reward: Normalisiertes BIP-Wachstum + Stabilitäts-Penalty
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, num_firms=10, num_households=50, render_mode=None):
        super().__init__()
        
        self.num_firms = num_firms
        self.num_households = num_households
        self.render_mode = render_mode
        
        # Observation Space: [BIP, Inflation, Arbeitslosigkeit, Durchschnittslohn, Kapital_gesamt]
        self.observation_space = spaces.Box(
            low=np.array([0, -0.5, 0, 0, 0], dtype=np.float32),
            high=np.array([10000, 0.5, 1.0, 1000, 100000], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action Space (für Regierung): [Steuersatz (0-0.5), Staatsausgaben (0-1000), Zinssatz (0-0.2)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.5, 1000.0, 0.2], dtype=np.float32),
            dtype=np.float32
        )
        
        # Wirtschafts-Zustand
        self.bip = 0.0
        self.inflation = 0.0
        self.unemployment = 0.0
        self.avg_wage = 50.0
        self.total_capital = 0.0
        
        # Firmen-Zustand (vereinfacht, regelbasiert für v1)
        self.firm_prices = np.random.uniform(10, 20, num_firms)
        self.firm_wages = np.random.uniform(40, 60, num_firms)
        self.firm_employees = np.random.randint(3, 8, num_firms)
        self.firm_capital = np.random.uniform(500, 1500, num_firms)
        
        # Haushalts-Zustand (regelbasiert)
        self.household_savings = np.random.uniform(100, 500, num_households)
        self.household_employed = np.random.choice([True, False], num_households, p=[0.95, 0.05])
        
        self.timestep = 0
        self.max_steps = 100
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset Wirtschaft
        self.bip = 1000.0
        self.inflation = 0.02
        self.unemployment = 0.05
        self.avg_wage = 50.0
        self.total_capital = np.sum(self.firm_capital)
        
        self.timestep = 0
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        # Action: [Steuersatz, Staatsausgaben, Zinssatz]
        tax_rate = np.clip(action[0], 0.0, 0.5)
        gov_spending = np.clip(action[1], 0.0, 1000.0)
        interest_rate = np.clip(action[2], 0.0, 0.2)
        
        # 1. Firmen-Produktion (vereinfacht)
        total_production = 0.0
        for i in range(self.num_firms):
            # Produktion = Mitarbeiter * Produktivität * Kapital-Faktor
            productivity = 20.0
            capital_factor = 1.0 + (self.firm_capital[i] / 1000.0) * 0.1
            production = self.firm_employees[i] * productivity * capital_factor
            total_production += production
            
            # Kosten: Löhne
            labor_cost = self.firm_employees[i] * self.firm_wages[i]
            
            # Profit (vereinfacht)
            revenue = production * self.firm_prices[i]
            profit = revenue - labor_cost - (self.firm_capital[i] * interest_rate)
            
            # Kapital-Update
            self.firm_capital[i] = max(0, self.firm_capital[i] + profit * 0.1)
        
        # 2. Haushalts-Konsum
        total_consumption = 0.0
        employed_count = 0
        for i in range(self.num_households):
            if self.household_employed[i]:
                # Einkommen = Durchschnittslohn
                income = self.avg_wage
                employed_count += 1
            else:
                income = gov_spending / self.num_households  # Arbeitslosengeld
            
            # Konsum = 80% Einkommen, 20% Sparen
            consumption = income * 0.8
            savings = income * 0.2
            
            self.household_savings[i] += savings
            total_consumption += consumption
        
        # 3. Makro-Indikatoren berechnen
        prev_bip = self.bip
        self.bip = total_production * 0.8 + total_consumption * 0.2  # Vereinfachtes BIP
        self.total_capital = np.sum(self.firm_capital)
        self.avg_wage = np.mean(self.firm_wages)
        self.unemployment = 1.0 - (employed_count / self.num_households)
        
        # Inflation (vereinfacht): Preis-Änderung
        avg_price = np.mean(self.firm_prices)
        self.inflation = (avg_price - 15.0) / 15.0 * 0.1  # Ziel-Preis: 15
        
        # 4. Reward-Berechnung (für Regierung)
        bip_growth = (self.bip - prev_bip) / max(prev_bip, 1.0)
        
        # Reward: BIP-Wachstum + Stabilitäts-Bonus - Penalty für hohe Arbeitslosigkeit
        reward = bip_growth * 100.0  # Scale für RL
        reward -= abs(self.inflation) * 50.0  # Inflations-Penalty
        reward -= self.unemployment * 200.0  # Arbeitslosigkeits-Penalty
        reward -= abs(tax_rate - 0.3) * 10.0  # Penalty für extreme Steuern
        
        # Normalisierung: -20 bis +10
        reward = np.clip(reward, -20.0, 10.0)
        
        # 5. Episode Ende?
        self.timestep += 1
        terminated = self.timestep >= self.max_steps
        truncated = False
        
        # Crash-Bedingungen
        if self.bip < 100 or self.unemployment > 0.5:
            terminated = True
            reward = -20.0  # Crash-Penalty
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        return np.array([
            self.bip,
            self.inflation,
            self.unemployment,
            self.avg_wage,
            self.total_capital
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            "bip": float(self.bip),
            "inflation": float(self.inflation),
            "unemployment": float(self.unemployment),
            "avg_wage": float(self.avg_wage),
            "timestep": self.timestep
        }
    
    def render(self):
        if self.render_mode == "human":
            print(f"\n=== Timestep {self.timestep} ===")
            print(f"BIP: {self.bip:.2f} | Inflation: {self.inflation:.3f} | Arbeitslosigkeit: {self.unemployment:.2%}")
            print(f"Durchschnittslohn: {self.avg_wage:.2f} | Gesamt-Kapital: {self.total_capital:.2f}")
