"""
移動平均 (SMA/EMA) 指標
"""

import pandas as pd
import numpy as np


def calculate_sma(close: pd.Series, period: int = 20) -> pd.Series:
    """
    計算簡引移動平均 (Simple Moving Average)
    
    Args:
        close: 收盤價時間序列
        period: 窗口期間 (預設 20)
    
    Returns:
        pd.Series: SMA 值
    """
    return close.rolling(window=period).mean()


def calculate_ema(close: pd.Series, period: int = 20) -> pd.Series:
    """
    計算指數移動平均 (Exponential Moving Average)
    
    Args:
        close: 收盤價時間序列
        period: 窗口期間 (預設 20)
    
    Returns:
        pd.Series: EMA 值
    """
    return close.ewm(span=period, adjust=False).mean()
