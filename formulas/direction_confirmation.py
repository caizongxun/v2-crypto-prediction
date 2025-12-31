"""
方向確認公式
"""

import pandas as pd
import numpy as np
from typing import Dict


class DirectionConfirmationFormula:
    """
    方向確認公式
    
    輸出: 0 ~ 1 (> 0.5 = 看多, < 0.5 = 看空, = 0.5 = 中性)
    描述: 買賣方評估值
    """
    
    def __init__(self):
        pass
    
    def generate_random_coefficients(self) -> Dict:
        """
        隨機生成公式係數
        """
        return {
            'w_rsi_direction': np.random.uniform(-1, 1),
            'w_macd_direction': np.random.uniform(-1, 1),
            'w_price_position': np.random.uniform(-1, 1),
            'w_ema_slope': np.random.uniform(-1, 1),
            'w_stochastic_direction': np.random.uniform(-1, 1),
            'bias': np.random.uniform(-0.5, 0.5)
        }
    
    def calculate(self, indicators: Dict, coefficients: Dict) -> pd.Series:
        """
        計算方向確認
        
        Args:
            indicators: 指標字典
            coefficients: 係數
        
        Returns:
            pd.Series: 方向確認 (0-1)
        """
        
        # RSI 方向
        rsi = indicators.get('rsi', pd.Series(50))
        rsi_direction = (rsi - 50) / 50  # -1 ~ 1
        rsi_dir_norm = (rsi_direction + 1) / 2  # 0 ~ 1
        
        # MACD 方向
        macd_line = indicators.get('macd_line', pd.Series(0))
        signal_line = indicators.get('signal_line', pd.Series(0))
        macd_diff = macd_line - signal_line
        macd_direction = np.sign(macd_diff)
        macd_dir_norm = (macd_direction + 1) / 2  # -1 => 0, 0 => 0.5, 1 => 1
        
        # 價格位置 (BB 中瘹算了，目前價格位置)
        close = indicators.get('close', pd.Series(100))
        upper_band = indicators.get('upper_band', pd.Series(100))
        lower_band = indicators.get('lower_band', pd.Series(100))
        
        price_position = (close - lower_band) / (upper_band - lower_band + 1e-10)
        price_position = np.clip(price_position, 0, 1).fillna(0.5)
        
        # EMA 斜率 (快速 EMA 基於緩慢 EMA 的斋律)
        ema_fast = indicators.get('ema_fast', pd.Series(100))
        ema_slow = indicators.get('ema_slow', pd.Series(100))
        ema_slope = ema_fast - ema_slow
        ema_slope_norm = np.clip(np.sign(ema_slope) * 0.5 + 0.5, 0, 1)
        
        # Stochastic 方向
        k_line = indicators.get('k_line', pd.Series(50))
        d_line = indicators.get('d_line', pd.Series(50))
        stoch_direction = k_line - d_line
        stoch_dir_norm = np.clip(np.sign(stoch_direction) * 0.5 + 0.5, 0, 1)
        
        # 統合公式
        score = (
            coefficients['w_rsi_direction'] * rsi_dir_norm +
            coefficients['w_macd_direction'] * macd_dir_norm +
            coefficients['w_price_position'] * price_position +
            coefficients['w_ema_slope'] * ema_slope_norm +
            coefficients['w_stochastic_direction'] * stoch_dir_norm +
            coefficients['bias']
        )
        
        # 正見化到 0-1
        score = (score - score.min()) / (score.max() - score.min() + 1e-10)
        score = score.fillna(0.5)
        score = np.clip(score, 0, 1)
        
        return score
