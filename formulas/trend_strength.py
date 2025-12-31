"""
趨勢強度公式
"""

import pandas as pd
import numpy as np
from typing import Dict


class TrendStrengthFormula:
    """
    趨勢強度公式
    
    輸出: 0 ~ 1 (程度逐漏的值提示趨勢的強度)
    描述: 日後教是一起蠈7日的準確率
    """
    
    def __init__(self):
        pass
    
    def generate_random_coefficients(self) -> Dict:
        """
        隨機生成公式係數
        """
        return {
            'w_rsi': np.random.uniform(-1, 1),
            'w_macd': np.random.uniform(-1, 1),
            'w_ema_diff': np.random.uniform(-1, 1),
            'w_atr_ratio': np.random.uniform(-1, 1),
            'w_volume': np.random.uniform(-1, 1),
            'bias': np.random.uniform(-0.5, 0.5)
        }
    
    def calculate(self, indicators: Dict, coefficients: Dict) -> pd.Series:
        """
        計算趨勢強度
        
        Args:
            indicators: 指標字典
            coefficients: 係數
        
        Returns:
            pd.Series: 趨勢強度 (0-1)
        """
        
        # RSI 正视
        rsi = indicators.get('rsi', pd.Series(0))
        rsi_norm = rsi / 100  # 正見化 0-1
        
        # MACD 統計
        macd_histogram = indicators.get('macd_histogram', pd.Series(0))
        macd_norm = np.sign(macd_histogram) * np.abs(macd_histogram).clip(0, 1)
        
        # EMA 下斧
        ema_fast = indicators.get('ema_fast', pd.Series(0))
        ema_slow = indicators.get('ema_slow', pd.Series(0))
        ema_diff = (ema_fast - ema_slow) / (ema_slow + 1e-10)
        ema_diff_norm = np.clip(np.abs(ema_diff), 0, 1)
        
        # ATR 比率
        atr = indicators.get('atr', pd.Series(1))
        close = indicators.get('close', pd.Series(1))
        atr_ratio = (atr / (close + 1e-10)).clip(0, 1)
        
        # 成交量
        volume_ratio = indicators.get('volume_ratio', pd.Series(1)).clip(0, 2)
        volume_norm = np.minimum(volume_ratio, 1)
        
        # 統合公式
        score = (
            coefficients['w_rsi'] * rsi_norm +
            coefficients['w_macd'] * macd_norm +
            coefficients['w_ema_diff'] * ema_diff_norm +
            coefficients['w_atr_ratio'] * atr_ratio +
            coefficients['w_volume'] * volume_norm +
            coefficients['bias']
        )
        
        # 正見化到 0-1
        score = (score - score.min()) / (score.max() - score.min() + 1e-10)
        score = score.fillna(0.5)
        score = np.clip(score, 0, 1)
        
        return score
