"""
Moving Average Convergence Divergence (MACD) 指標
"""

import pandas as pd
import numpy as np


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple:
    """
    計算 MACD (移動平均匯聚不光離)
    
    Args:
        close: 收盤價時間序列
        fast: 快速移動平均期間 (預設 12)
        slow: 緩慢移動平均期間 (預設 26)
        signal: 信號線期間 (預設 9)
    
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    # 計算指數移動平均
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    # MACD 線
    macd_line = ema_fast - ema_slow
    
    # 信號線 (信號線是 MACD 的移動平均)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # 直方圖 (Histogram)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram
