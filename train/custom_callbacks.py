"""Custom Callbacks to log detailed environment metrics to TensorBoard"""

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from typing import Dict, Optional


class EconomyMetricsCallbacks(DefaultCallbacks):
    """Custom callbacks to log economy-specific metrics"""
    
    def on_episode_end(
        self,
        *,
        worker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: Optional[int] = None,
        **kwargs
    ) -> None:
        """Called when an episode ends"""
        
        # Get episode info (enthält custom metrics vom Environment)
        info = episode.last_info_for()
        
        if info:
            # Economy-wide Metrics
            if 'bankrupt_firms' in info:
                episode.custom_metrics['economy/bankrupt_firms'] = info['bankrupt_firms']
            if 'bankrupt_households' in info:
                episode.custom_metrics['economy/bankrupt_households'] = info['bankrupt_households']
            
            # Firm Metrics (wenn vorhanden)
            if 'avg_firm_price' in info:
                episode.custom_metrics['firms/avg_price'] = info['avg_firm_price']
            if 'avg_firm_wage' in info:
                episode.custom_metrics['firms/avg_wage'] = info['avg_firm_wage']
            if 'avg_firm_production' in info:
                episode.custom_metrics['firms/avg_production'] = info['avg_firm_production']
            if 'avg_firm_inventory' in info:
                episode.custom_metrics['firms/avg_inventory'] = info['avg_firm_inventory']
            
            # Household Metrics (wenn vorhanden)
            if 'avg_household_consumption' in info:
                episode.custom_metrics['households/avg_consumption'] = info['avg_household_consumption']
            if 'avg_household_wealth' in info:
                episode.custom_metrics['households/avg_wealth'] = info['avg_household_wealth']
            if 'avg_household_utility' in info:
                episode.custom_metrics['households/avg_utility'] = info['avg_household_utility']
            
            # Market Metrics
            if 'total_transactions' in info:
                episode.custom_metrics['market/total_transactions'] = info['total_transactions']
            if 'avg_market_price' in info:
                episode.custom_metrics['market/avg_price'] = info['avg_market_price']
    
    def on_episode_step(
        self,
        *,
        worker,
        base_env: BaseEnv,
        policies: Optional[Dict[str, Policy]] = None,
        episode: Episode,
        env_index: Optional[int] = None,
        **kwargs
    ) -> None:
        """Called on each episode step (optional - for step-wise metrics)"""
        pass

    def on_train_result(
        self,
        *,
        algorithm,
        result: dict,
        **kwargs
    ) -> None:
        """Called after training iteration"""
        
        # Aggregate custom metrics across all workers
        if 'custom_metrics' in result:
            custom = result['custom_metrics']
            
            # Log to result dict (wird automatisch zu TensorBoard gesendet)
            for key, value in custom.items():
                if isinstance(value, (int, float)):
                    result[f'custom/{key}'] = value
