"""Multi-Agent Wirtschafts-Environment (BASIC VERSION)

Simple Simulation mit:
- 10 Haushalten (konsumieren)
- 5 Firmen (produzieren)
- Kein Staat (erstmal)

Jeder Agent ist RL-gesteuert.
Fixe Startbedingungen aus Config-Files.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import yaml
import os


class MultiAgentEconomyEnv(gym.Env):
    """Basic Multi-Agent Wirtschaft"""
    
    metadata = {'render_modes': []}
    
    def __init__(self, config_path='configs'):
        super().__init__()
        
        # Load Configs
        self._load_configs(config_path)
        
        # Zeitstruktur
        self.steps_per_episode = 250  # 1 Jahr
        self.current_step = 0
        self.episode_count = 0
        
        # Observation & Action Spaces (multi-agent)
        self._setup_spaces()
        
        # Initiale Bedingungen speichern (fix über alle Episoden!)
        self.initial_households = self.household_config['households']['initial_conditions']
        self.initial_firms = self.firm_config['firms']['initial_conditions']
        
        # Aktuelle Zustände
        self.households = []
        self.firms = []
        
    def _load_configs(self, config_path):
        """Lade YAML Configs"""
        with open(os.path.join(config_path, 'household_config.yaml')) as f:
            self.household_config = yaml.safe_load(f)
        with open(os.path.join(config_path, 'firm_config.yaml')) as f:
            self.firm_config = yaml.safe_load(f)
    
    def _setup_spaces(self):
        """Definiere Observation & Action Spaces"""
        # HAUSHALTE: Observation = [eigenes Cash, Durchschnittspreis, Arbeitslosigkeit]
        self.household_obs_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([10000, 50, 1], dtype=np.float32)
        )
        
        # HAUSHALTE: Action = [Konsumquote 0-100%]
        self.household_action_space = spaces.Box(
            low=np.array([0.5], dtype=np.float32),  # Min 50%
            high=np.array([1.0], dtype=np.float32),  # Max 100%
        )
        
        # FIRMEN: Observation = [eigenes Kapital, eigener Lagerbestand, Durchschnittsnachfrage]
        self.firm_obs_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1000000, 1000, 500], dtype=np.float32)
        )
        
        # FIRMEN: Action = [Preis, Produktion]
        self.firm_action_space = spaces.Box(
            low=np.array([5, 0], dtype=np.float32),    # Min Preis 5€, keine Produktion
            high=np.array([20, 200], dtype=np.float32)  # Max Preis 20€, max 200 Einheiten
        )
        
        # Kombinierter Space (für Single-Agent Wrapper später)
        # Erstmal: Jeder Agent separat
    
    def reset(self, seed=None, options=None):
        """Zurück zu FIXEN Startbedingungen"""
        super().reset(seed=seed)
        
        self.episode_count += 1
        self.current_step = 0
        
        # Haushalte mit FIXEN Werten aus Config
        self.households = []
        for h_config in self.initial_households:
            self.households.append({
                'id': h_config['id'],
                'cash': float(h_config['cash']),  # IMMER gleich!
                'type': h_config['type'],
                'employed': True,
                'bankrupt': False
            })
        
        # Firmen mit FIXEN Werten aus Config
        self.firms = []
        firm_params = self.firm_config['firms']['params']
        for f_config in self.initial_firms:
            self.firms.append({
                'id': f_config['id'],
                'capital': float(f_config['capital']),    # IMMER gleich!
                'employees': f_config['employees'],       # IMMER gleich!
                'type': f_config['type'],
                'inventory': float(firm_params['initial_inventory']),
                'price': float(firm_params['initial_price']),
                'bankrupt': False
            })
        
        print(f"\n📅 Episode {self.episode_count} | Tag 1/250")
        
        return self._get_observations(), self._get_info()
    
    def step(self, actions):
        """
        Simuliere 1 Tag
        
        Args:
            actions: Dict mit allen Agent-Actions
                     {'household_0': [0.8], 'household_1': [0.75], ..., 
                      'firm_0': [12, 80], 'firm_1': [10, 100], ...}
        """
        self.current_step += 1
        
        # 1. Firmen produzieren
        self._firms_produce(actions)
        
        # 2. Haushalte konsumieren
        self._households_consume(actions)
        
        # 3. Markt: Haushalte kaufen von Firmen
        self._market_clearing()
        
        # 4. Firmen zahlen Löhne
        self._firms_pay_wages()
        
        # 5. Pleite-Check
        self._check_bankruptcy()
        
        # 6. Rewards berechnen
        rewards = self._compute_rewards()
        
        # Episode Ende?
        terminated = (self.current_step >= self.steps_per_episode)
        truncated = False
        
        if self.current_step % 50 == 0:
            print(f"   Tag {self.current_step}/250")
        
        return self._get_observations(), rewards, terminated, truncated, self._get_info()
    
    def _firms_produce(self, actions):
        """Firmen produzieren basierend auf Actions"""
        firm_params = self.firm_config['firms']['params']
        
        for i, firm in enumerate(self.firms):
            if firm['bankrupt']:
                continue
            
            # Action: [Preis, Produktionsmenge]
            firm_action = actions.get(f'firm_{i}', [10, 50])  # Default
            price, production = firm_action[0], firm_action[1]
            
            firm['price'] = float(price)
            
            # Produktionskosten (vereinfacht)
            production_cost = production * 2  # 2€ pro Einheit
            
            if firm['capital'] >= production_cost:
                firm['inventory'] += production
                firm['capital'] -= production_cost
    
    def _households_consume(self, actions):
        """Haushalte entscheiden über Konsum"""
        for i, household in enumerate(self.households):
            if household['bankrupt']:
                continue
            
            # Action: [Konsumquote]
            h_action = actions.get(f'household_{i}', [0.8])  # Default: 80%
            consumption_rate = h_action[0]
            
            # Konsumieren
            consumption_amount = household['cash'] * consumption_rate
            household['planned_consumption'] = consumption_amount
    
    def _market_clearing(self):
        """Haushalte kaufen von Firmen"""
        # Durchschnittspreis
        active_firms = [f for f in self.firms if not f['bankrupt']]
        if not active_firms:
            return
        
        avg_price = np.mean([f['price'] for f in active_firms])
        
        # Gesamte Nachfrage
        total_demand = sum(h.get('planned_consumption', 0) for h in self.households)
        
        # Firmen verkaufen (proportional zu Inventory)
        total_inventory = sum(f['inventory'] for f in active_firms)
        
        for firm in active_firms:
            if total_inventory > 0:
                firm_share = firm['inventory'] / total_inventory
                revenue = total_demand * firm_share
                
                # Verkaufte Menge
                sold = min(firm['inventory'], revenue / firm['price'])
                
                firm['capital'] += revenue
                firm['inventory'] -= sold
                firm['revenue'] = revenue  # Für Reward-Berechnung
    
    def _firms_pay_wages(self):
        """Firmen zahlen Löhne an Haushalte"""
        firm_params = self.firm_config['firms']['params']
        wage = firm_params['wage_per_employee']
        
        active_households = [h for h in self.households if not h['bankrupt']]
        if not active_households:
            return
        
        for firm in self.firms:
            if firm['bankrupt']:
                continue
            
            total_wages = firm['employees'] * wage
            
            if firm['capital'] >= total_wages:
                firm['capital'] -= total_wages
                
                # Verteile Löhne an Haushalte (gleichmäßig)
                wage_per_household = total_wages / len(active_households)
                for household in active_households:
                    household['cash'] += wage_per_household
    
    def _check_bankruptcy(self):
        """Prüfe ob Agents pleite sind"""
        for household in self.households:
            if household['cash'] < 0 and not household['bankrupt']:
                household['bankrupt'] = True
                print(f"   💥 Haushalt {household['id']} ({household['type']}) pleite!")
        
        for firm in self.firms:
            if firm['capital'] < 0 and not firm['bankrupt']:
                firm['bankrupt'] = True
                print(f"   💥 Firma {firm['id']} ({firm['type']}) pleite!")
    
    def _compute_rewards(self):
        """Rewards für alle Agents"""
        rewards = {}
        
        # Haushalte: Konsum ist gut, Pleite ist schlecht
        for i, household in enumerate(self.households):
            if household['bankrupt']:
                rewards[f'household_{i}'] = -100
            else:
                # Reward basierend auf verbleibendem Cash
                rewards[f'household_{i}'] = household['cash'] / 1000  # Normalisiert
        
        # Firmen: Gewinn ist gut, Pleite ist schlecht
        for i, firm in enumerate(self.firms):
            if firm['bankrupt']:
                rewards[f'firm_{i}'] = -100
            else:
                # Reward basierend auf Kapital
                profit = firm.get('revenue', 0) - (firm['employees'] * 50)  # Revenue - Kosten
                rewards[f'firm_{i}'] = profit / 1000  # Normalisiert
        
        return rewards
    
    def _get_observations(self):
        """Observations für alle Agents"""
        observations = {}
        
        # Markt-Info
        active_firms = [f for f in self.firms if not f['bankrupt']]
        avg_price = np.mean([f['price'] for f in active_firms]) if active_firms else 10
        unemployment = sum(1 for h in self.households if h['bankrupt']) / len(self.households)
        
        # Haushalte
        for i, household in enumerate(self.households):
            observations[f'household_{i}'] = np.array([
                household['cash'],
                avg_price,
                unemployment
            ], dtype=np.float32)
        
        # Firmen
        total_demand = sum(h.get('planned_consumption', 0) for h in self.households)
        avg_demand = total_demand / len(active_firms) if active_firms else 0
        
        for i, firm in enumerate(self.firms):
            observations[f'firm_{i}'] = np.array([
                firm['capital'],
                firm['inventory'],
                avg_demand
            ], dtype=np.float32)
        
        return observations
    
    def _get_info(self):
        """Zusätzliche Infos"""
        active_households = sum(1 for h in self.households if not h['bankrupt'])
        active_firms = sum(1 for f in self.firms if not f['bankrupt'])
        
        return {
            'step': self.current_step,
            'episode': self.episode_count,
            'active_households': active_households,
            'active_firms': active_firms,
            'total_cash': sum(h['cash'] for h in self.households),
            'total_capital': sum(f['capital'] for f in self.firms)
        }


# Gymnasium Registration
from gymnasium.envs.registration import register

register(
    id='MultiAgentEconomy-v0',
    entry_point='envs.multi_agent_economy:MultiAgentEconomyEnv',
)
