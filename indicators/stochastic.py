"""
Stochastic Oscillator 隊機絷迼粗顊指標
"""

import pandas as pd
import numpy as np


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> tuple:
    """
    計算隊機絷迼粗顊 (Stochastic Oscillator)
    
    Args:
        high: 最高價時間序列
        low: 最低價時間序列
        close: 收盤價時間序列
        period: 窗口期間 (預設 14)
        smooth_k: K 線平滑 (預設 3)
        smooth_d: D 線平滑 (預設 3)
    
    Returns:
        tuple: (k_line, d_line)
    """
    # 最高最低
    low_min = low.rolling(window=period).min()
    high_max = high.rolling(window=period).max()
    
    # 粗顊 %K
    k_percent = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    
    # 平滑 %K (很多平台稱为 %K)
    k_line = k_percent.rolling(window=smooth_k).mean()
    
    # %D 是 %K 的移動平均
    d_line = k_line.rolling(window=smooth_d).mean()
    
    return k_line, d_line
