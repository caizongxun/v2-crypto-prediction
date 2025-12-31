"""
Relative Strength Index (RSI) 指標
"""

import pandas as pd
import numpy as np


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    計算 RSI (相對強劣指數)
    
    Args:
        close: 收盤價時間序列
        period: 窗口期間 (預設 14)
    
    Returns:
        pd.Series: RSI 值 (0-100)
    """
    # 計算往後的價格變化
    delta = close.diff()
    
    # 分離正值和負值的暯幅
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # 避免除以零错誤
    rs = gain / (loss + 1e-10)
    
    # 計算 RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
