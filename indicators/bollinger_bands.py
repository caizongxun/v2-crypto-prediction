"""
Bollinger Bands 布林帶
"""

import pandas as pd
import numpy as np


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple:
    """
    計算布林帶 (Bollinger Bands)
    
    Args:
        close: 收盤價時間序列
        period: 窗口期間 (預設 20)
        num_std: 標準差倍數 (預設 2.0)
    
    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    # 中佋線 (移動平均)
    middle_band = close.rolling(window=period).mean()
    
    # 標準差
    std = close.rolling(window=period).std()
    
    # 上下趨勢線
    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)
    
    return middle_band, upper_band, lower_band
