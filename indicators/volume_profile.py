"""
成交量指標
"""

import pandas as pd
import numpy as np


def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    計算成交量移動平均
    
    Args:
        volume: 成交量時間序列
        period: 窗口期間 (預設 20)
    
    Returns:
        pd.Series: 成交量 MA
    """
    return volume.rolling(window=period).mean()


def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    計算成交量比率 (當前成交量 / 平均成交量)
    
    Args:
        volume: 成交量時間序列
        period: 窗口期間 (預設 20)
    
    Returns:
        pd.Series: 成交量比率
    """
    vol_ma = calculate_volume_sma(volume, period)
    return volume / (vol_ma + 1e-10)
