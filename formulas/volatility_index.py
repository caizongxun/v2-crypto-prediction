"""
波動率公式
"""

import pandas as pd
import numpy as np
from typing import Dict


class VolatilityIndexFormula:
    """
    波動率公式
    
    輸出: 0 ~ 1 (波動的求平此數)
    描述: 市場的不穩定性
    """
    
    def __init__(self):
        pass
    
    def generate_random_coefficients(self) -> Dict:
        """
        隨機生成公式係數
        """
        return {
            'w_atr': np.random.uniform(-1, 1),
            'w_bollinger_width': np.random.uniform(-1, 1),
            'w_volume_volatility': np.random.uniform(-1, 1),
            'w_price_change': np.random.uniform(-1, 1),
            'w_stochastic_range': np.random.uniform(-1, 1),
            'bias': np.random.uniform(-0.5, 0.5)
        }
    
    def calculate(self, indicators: Dict, coefficients: Dict) -> pd.Series:
        """
        計算波動率
        
        Args:
            indicators: 指標字典
            coefficients: 係數
        
        Returns:
            pd.Series: 波動率 (0-1)
        """
        
        # ATR 正見化
        atr = indicators.get('atr', pd.Series(1))
        close = indicators.get('close', pd.Series(1))
        atr_norm = (atr / (close + 1e-10)).clip(0, 1)
        
        # 布林带寶寬度
        upper_band = indicators.get('upper_band', pd.Series(100))
        lower_band = indicators.get('lower_band', pd.Series(100))
        middle_band = indicators.get('middle_band', pd.Series(100))
        
        bb_width = (upper_band - lower_band) / (middle_band + 1e-10)
        bb_width_norm = (bb_width / bb_width.max()).clip(0, 1).fillna(0.5)
        
        # 成交量波動
        volume = indicators.get('volume', pd.Series(1))
        volume_sma = volume.rolling(window=20).mean()
        volume_volatility = (volume / (volume_sma + 1e-10)).std()
        volume_vol_norm = np.minimum(volume_volatility, 2) / 2
        
        # 價格變化
        price_change = close.pct_change().abs()
        price_change_norm = (price_change / price_change.max()).clip(0, 1).fillna(0.5)
        
        # Stochastic 範圍
        k_line = indicators.get('k_line', pd.Series(50))
        d_line = indicators.get('d_line', pd.Series(50))
        stoch_range = (np.abs(k_line - d_line) / 100).clip(0, 1)
        
        # 統合公式
        score = (
            coefficients['w_atr'] * atr_norm +
            coefficients['w_bollinger_width'] * bb_width_norm +
            coefficients['w_volume_volatility'] * volume_vol_norm +
            coefficients['w_price_change'] * price_change_norm +
            coefficients['w_stochastic_range'] * stoch_range +
            coefficients['bias']
        )
        
        # 正見化到 0-1
        score = (score - score.min()) / (score.max() - score.min() + 1e-10)
        score = score.fillna(0.5)
        score = np.clip(score, 0, 1)
        
        return score
