"""Volkswirtschafts-Environment für Reinforcement Learning

Dieses Environment simuliert eine vereinfachte Volkswirtschaft mit:
- 10 Firmen (produzieren, setzen Preise, zahlen Löhne)
- 50 Haushalten (konsumieren, sparen)
- 1 Regierung (Steuern, Staatsausgaben, Zinspolitik)

Zeitstruktur:
- 1 Step = 1 Tag
- 1 Episode = 365 Tage = 1 Jahr
- Training über mehrere Jahre (Episoden)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EconomyEnv(gym.Env):
    """Gymnasium-kompatibles Wirtschafts-Environment
    
    Der RL-Agent steuert die Regierung.
    Firmen und Haushalte sind (vorerst) regelbasiert.
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, max_years=5):
        super().__init__()
        
        # === ZEITSTRUKTUR ===
        self.days_per_year = 365
        self.max_years = max_years
        self.max_steps = self.days_per_year * self.max_years
        self.current_day = 0
        self.current_year = 1
        self.current_step = 0
        
        # Jahresmetriken (für Aggregation)
        self.yearly_metrics = {
            'bip': [],
            'inflation': [],
            'unemployment': [],
            'deficit': []
        }
        
        # === WIRTSCHAFTS-AKTEURE ===
        self.num_firms = 10
        self.num_households = 50
        
        # Observation Space: [BIP, Inflation, Arbeitslosigkeit, Staatsschulden, Zinssatz]
        self.observation_space = spaces.Box(
            low=np.array([0, -0.5, 0, 0, 0], dtype=np.float32),
            high=np.array([100000, 0.5, 1.0, 50000, 0.3], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action Space: [Steuersatz, Staatsausgaben, Zinssatz]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.5, 1000.0, 0.2], dtype=np.float32),
            dtype=np.float32
        )
        
        # Wirtschafts-Zustand
        self.firms = []
        self.households = []
        self.government = {
            'debt': 0.0,
            'deficit': 0.0,
            'tax_revenue': 0.0,
            'spending': 0.0
        }
        
        # Makro-Variablen
        self.bip = 0.0
        self.bip_last = 0.0
        self.inflation = 0.0
        self.interest_rate = 0.05
        self.avg_price_last = 0.0
        
    def reset(self, seed=None, options=None):
        """Reset Environment zu Anfangszustand"""
        super().reset(seed=seed)
        
        # Zeit zurücksetzen
        self.current_day = 0
        self.current_year = 1
        self.current_step = 0
        self.yearly_metrics = {'bip': [], 'inflation': [], 'unemployment': [], 'deficit': []}
        
        # Firmen initialisieren
        self.firms = [
            {
                'capital': 1000.0 + np.random.randn() * 100,
                'employees': 5,
                'price': 10.0,
                'inventory': 50,
                'revenue': 0.0,
                'costs': 0.0
            }
            for _ in range(self.num_firms)
        ]
        
        # Haushalte initialisieren
        self.households = [
            {
                'cash': 500.0 + np.random.randn() * 50,
                'income': 50.0,
                'employed': True
            }
            for _ in range(self.num_households)
        ]
        
        # Regierung zurücksetzen
        self.government = {
            'debt': 1000.0,
            'deficit': 0.0,
            'tax_revenue': 0.0,
            'spending': 500.0
        }
        
        # Makro-Variablen
        self.bip = 5000.0
        self.bip_last = 5000.0
        self.inflation = 0.02
        self.interest_rate = 0.05
        self.avg_price_last = 10.0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Simuliere einen Tag (1 Step)"""
        # Action clipping (Sicherheit)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        tax_rate, gov_spending, interest_rate = action
        
        self.interest_rate = float(interest_rate)
        
        # === 1. FIRMEN: Produzieren & Preise setzen ===
        total_production = 0
        total_labor_cost = 0
        
        for firm in self.firms:
            # Produktion (10 Einheiten pro Mitarbeiter pro Tag)
            production = firm['employees'] * 10
            firm['inventory'] += production
            total_production += production
            
            # Löhne zahlen (50€ pro Mitarbeiter pro Tag)
            labor_cost = firm['employees'] * 50
            firm['capital'] -= labor_cost
            firm['costs'] = labor_cost
            total_labor_cost += labor_cost
        
        # === 2. HAUSHALTE: Einkommen & Konsum ===
        total_consumption = 0
        employed_count = 0
        
        for household in self.households:
            if household['employed']:
                # Einkommen aus Arbeit
                household['cash'] += household['income']
                employed_count += 1
            
            # Konsum (80% vom Cash)
            consumption = household['cash'] * 0.8
            household['cash'] -= consumption
            total_consumption += consumption
        
        # === 3. MARKT: Firmen verkaufen an Haushalte ===
        avg_price = np.mean([f['price'] for f in self.firms])
        quantity_demanded = total_consumption / avg_price if avg_price > 0 else 0
        
        # Revenue an Firmen verteilen (proportional zu Inventory)
        total_inventory = sum(f['inventory'] for f in self.firms)
        
        for firm in self.firms:
            if total_inventory > 0:
                firm_share = firm['inventory'] / total_inventory
                revenue = total_consumption * firm_share
                firm['revenue'] = revenue
                firm['capital'] += revenue
                
                # Inventory reduzieren
                sold = min(firm['inventory'], quantity_demanded * firm_share)
                firm['inventory'] -= sold
        
        # === 4. REGIERUNG: Steuern & Ausgaben ===
        # Steuern von Firmen
        tax_revenue = sum(max(0, f['revenue'] - f['costs']) * tax_rate for f in self.firms)
        self.government['tax_revenue'] = tax_revenue
        
        # Staatsausgaben
        self.government['spending'] = float(gov_spending)
        
        # Defizit/Überschuss
        deficit = gov_spending - tax_revenue
        self.government['deficit'] = deficit
        self.government['debt'] += deficit
        
        # Staatsausgaben fließen in Wirtschaft (vereinfacht: an Haushalte)
        if self.num_households > 0:
            per_household = gov_spending / self.num_households
            for household in self.households:
                household['cash'] += per_household
        
        # === 5. MAKRO-VARIABLEN BERECHNEN ===
        # BIP (Vereinfacht: Gesamtproduktion * Durchschnittspreis)
        self.bip = total_production * avg_price
        
        # BIP-Wachstum
        bip_growth = (self.bip / self.bip_last - 1.0) if self.bip_last > 0 else 0.0
        self.bip_last = self.bip
        
        # Inflation (Preisänderung)
        self.inflation = (avg_price / self.avg_price_last - 1.0) if self.avg_price_last > 0 else 0.0
        self.avg_price_last = avg_price
        
        # Arbeitslosigkeit
        unemployment = 1.0 - (employed_count / self.num_households)
        
        # === 6. REWARD BERECHNEN ===
        reward = self._compute_reward(bip_growth, unemployment, self.inflation, deficit)
        
        # === 7. ZEIT FORTSCHREITEN ===
        self.current_step += 1
        self.current_day += 1
        
        # Jahreswechsel
        if self.current_day >= self.days_per_year:
            self._year_end_summary()
            self.current_day = 0
            self.current_year += 1
        
        # Episode Ende
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        obs = self._get_observation()
        info = self._get_info()
        info['bip_growth'] = bip_growth
        info['day'] = self.current_day
        info['year'] = self.current_year
        
        return obs, reward, terminated, truncated, info
    
    def _year_end_summary(self):
        """Jahresabschluss: Metriken sammeln"""
        self.yearly_metrics['bip'].append(self.bip)
        self.yearly_metrics['inflation'].append(self.inflation)
        
        employed = sum(1 for h in self.households if h['employed'])
        unemployment = 1.0 - (employed / self.num_households)
        self.yearly_metrics['unemployment'].append(unemployment)
        self.yearly_metrics['deficit'].append(self.government['deficit'])
        
        print(f"\n📅 Jahr {self.current_year} abgeschlossen:")
        print(f"   BIP: {self.bip:.0f}€")
        print(f"   Inflation: {self.inflation:.2%}")
        print(f"   Arbeitslosigkeit: {unemployment:.1%}")
        print(f"   Staatsdefizit: {self.government['deficit']:.0f}€")
    
    def _compute_reward(self, bip_growth, unemployment, inflation, deficit):
        """Reward-Funktion für Regierung
        
        Ziele:
        - BIP-Wachstum maximieren (Wohlstand)
        - Arbeitslosigkeit minimieren (sozial)
        - Inflation bei ~2% halten (Stabilität)
        - Staatsdefizit gering halten (Nachhaltigkeit)
        """
        reward = (
            + bip_growth * 10.0           # BIP-Wachstum belohnen
            - unemployment * 20.0         # Arbeitslosigkeit stark bestrafen
            - abs(inflation) * 15.0       # Inflation (egal ob +/-) bestrafen
            - abs(deficit) * 0.01         # Defizit leicht bestrafen
        )
        
        # Clipping (verhindert extreme Rewards)
        reward = np.clip(reward, -20.0, 10.0)
        
        return float(reward)
    
    def _get_observation(self):
        """Aktuellen State für RL-Agent"""
        employed_count = sum(1 for h in self.households if h['employed'])
        unemployment = 1.0 - (employed_count / self.num_households)
        
        obs = np.array([
            self.bip / 10000.0,              # Normalisiert
            self.inflation,                   # -0.5 bis 0.5
            unemployment,                     # 0 bis 1
            self.government['debt'] / 10000.0, # Normalisiert
            self.interest_rate                # 0 bis 0.2
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self):
        """Zusätzliche Infos (für Logging)"""
        employed_count = sum(1 for h in self.households if h['employed'])
        unemployment = 1.0 - (employed_count / self.num_households)
        
        return {
            'bip': float(self.bip),
            'inflation': float(self.inflation),
            'unemployment': float(unemployment),
            'debt': float(self.government['debt']),
            'deficit': float(self.government['deficit']),
            'interest_rate': float(self.interest_rate),
            'step': self.current_step,
            'year': self.current_year,
            'day': self.current_day
        }
