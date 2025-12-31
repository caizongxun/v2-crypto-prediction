"""
Average True Range (ATR) 指標
"""

import pandas as pd
import numpy as np


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    計算 ATR (平均真實波動幅度)
    
    Args:
        high: 最高價時間序列
        low: 最低價時間序列
        close: 收盤價時間序列
        period: 窗口期間 (預設 14)
    
    Returns:
        pd.Series: ATR 值
    """
    # 計算 True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    # 取最大值
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 計算 ATR (使用指數移動平均)
    atr = tr.ewm(span=period, adjust=False).mean()
    
    return atr
