"""
技術指標模組 - 計算 EMA, SuperTrend, ATR, RSI 等
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


class TechnicalIndicators:
    """
    技術指標計算器
    """
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """
        計算 EMA (指數移動平均)
        
        Args:
            data: 價格數據
            period: 時間間隔
        
        Returns:
            pd.Series: EMA 值
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        計算 ATR (平均真實波幅)
        
        ATR 測量價格波動率：
        - TR = max(high - low, high - close[t-1], close[t-1] - low)
        - ATR = EMA(TR, period)
        
        Args:
            high: 最高價
            low: 最低價
            close: 收盤價
            period: 時間間隔 (預設 14)
        
        Returns:
            pd.Series: ATR 值
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def calculate_supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        計算 SuperTrend (趨勢趨位指標)
        
        SuperTrend 是出趣段穋指標，結合了 ATR 和 HL2 的單位敷攫
        
        Args:
            high: 最高價
            low: 最低價
            close: 收盤價
            period: HL2 的時間間隔 (預設 10)
            multiplier: ATR 倍數 (預設 3.0)
        
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (supertrend, basic_upper, basic_lower)
        """
        hl2 = (high + low) / 2
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        
        # 基礎上下梎帶
        basic_ub = hl2 + multiplier * atr
        basic_lb = hl2 - multiplier * atr
        
        # 最終上下梎带
        final_ub = np.where(basic_ub < close.shift(1), basic_ub, close.shift(1))
        final_lb = np.where(basic_lb > close.shift(1), basic_lb, close.shift(1))
        
        # SuperTrend
        supertrend = np.where(close <= final_ub, final_ub, final_lb)
        
        return pd.Series(supertrend, index=close.index), pd.Series(basic_ub, index=close.index), pd.Series(basic_lb, index=close.index)
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        計算 RSI (相對強弱指數)
        
        RSI 測量上漲既絶的轉折點
        - RSI = 100 - (100 / (1 + RS))
        - RS = 上漲平均 / 下跌平均
        
        Args:
            close: 收盤價
            period: 時間間隔 (預設 14)
        
        Returns:
            pd.Series: RSI 值 (0-100)
        """
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)  # 預防除以零 區分 1e-10
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_stochastic_rsi(
        close: pd.Series,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        計算 Stochastic RSI (隨機 RSI)
        
        結合了 RSI 和 Stochastic Oscillator 的水平動掛潜力
        
        Args:
            close: 收盤價
            rsi_period: RSI 時間間隔
            stoch_period: Stochastic 時間間隔
            smooth_k: K 债平滑時間間隔
            smooth_d: D 债平滑時間間隔
        
        Returns:
            Tuple[pd.Series, pd.Series]: (K 债, D 债)
        """
        rsi = TechnicalIndicators.calculate_rsi(close, rsi_period)
        
        # Stochastic 載換
        rsi_low = rsi.rolling(window=stoch_period).min()
        rsi_high = rsi.rolling(window=stoch_period).max()
        
        k_value = 100 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)
        d_value = k_value.rolling(window=smooth_d).mean()
        
        # 平滑 K 债
        k_value_smooth = k_value.rolling(window=smooth_k).mean()
        
        return k_value_smooth, d_value
    
    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 12) -> pd.Series:
        """
        計算 ROC (價格動量) 變化率
        
        ROC 測量價格動量率
        - ROC = (close - close[t-period]) / close[t-period] * 100
        
        Args:
            close: 收盤價
            period: 時間間隔 (預設 12)
        
        Returns:
            pd.Series: ROC 值 (%)
        """
        return ((close - close.shift(period)) / close.shift(period)) * 100
    
    @staticmethod
    def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        計算成交量 SMA (粗算移動平均)
        
        Args:
            volume: 成交量
            period: 時間間隔
        
        Returns:
            pd.Series: 成交量 SMA
        """
        return volume.rolling(window=period).mean()
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        計算 VWAP (成交量加橫平均價格)
        
        VWAP = 累計(HL2 * volume) / 累計(volume)
        
        Args:
            high: 最高價
            low: 最低價
            close: 收盤價
            volume: 成交量
        
        Returns:
            pd.Series: VWAP 值
        """
        hl2 = (high + low) / 2
        vwap = (hl2 * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        return vwap
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        計算 ADX (平均趨澎指數)
        
        ADX 測量趨勢強度，不指方向。
        
        粗估技巧：使用 DI+/- 的折算
        
        Args:
            high: 最高價
            low: 最低價
            close: 收盤價
            period: 時間間隔 (預設 14)
        
        Returns:
            pd.Series: ADX 值 (0-100)
        """
        # 計算 DI
        plus_dm_raw = high.diff()
        minus_dm_raw = -low.diff()
        
        # 使用 np.where 休魚實數組
        plus_dm_arr = np.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), plus_dm_raw, 0)
        minus_dm_arr = np.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), minus_dm_raw, 0)
        
        # 載換回 Series
        plus_dm = pd.Series(plus_dm_arr, index=close.index)
        minus_dm = pd.Series(minus_dm_arr, index=close.index)
        
        # 計算 ATR
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        
        # 計算 DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        
        # 計算 ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        
        return adx


class IndicatorCalculator:
    """
    一體技術指標計算器
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化計算器
        
        Args:
            df: OHLCV 數據框
        """
        self.df = df.copy()
        self.indicators = {}
    
    def calculate_all_indicators(
        self,
        ema_periods: List[int] = [15, 60],
        atr_period: int = 14,
        rsi_period: int = 14,
        roc_period: int = 12,
        volume_period: int = 20
    ) -> pd.DataFrame:
        """
        計算所有指標
        
        Args:
            ema_periods: EMA 時間間隔 (預設 [15, 60])
            atr_period: ATR 時間間隔
            rsi_period: RSI 時間間隔
            roc_period: ROC 時間間隔
            volume_period: 成交量 SMA 時間間隔
        
        Returns:
            pd.DataFrame: 包含所有指標的 dataframe
        """
        # EMA
        for period in ema_periods:
            col_name = f"EMA_{period}"
            self.df[col_name] = TechnicalIndicators.calculate_ema(self.df['close'], period)
            self.indicators[col_name] = True
        
        # ATR
        self.df['ATR'] = TechnicalIndicators.calculate_atr(
            self.df['high'],
            self.df['low'],
            self.df['close'],
            atr_period
        )
        self.indicators['ATR'] = True
        
        # SuperTrend
        st, _, _ = TechnicalIndicators.calculate_supertrend(
            self.df['high'],
            self.df['low'],
            self.df['close']
        )
        self.df['SuperTrend'] = st
        self.indicators['SuperTrend'] = True
        
        # RSI
        self.df['RSI'] = TechnicalIndicators.calculate_rsi(self.df['close'], rsi_period)
        self.indicators['RSI'] = True
        
        # Stochastic RSI
        k_val, d_val = TechnicalIndicators.calculate_stochastic_rsi(self.df['close'])
        self.df['Stoch_RSI_K'] = k_val
        self.df['Stoch_RSI_D'] = d_val
        self.indicators['Stoch_RSI'] = True
        
        # ROC
        self.df['ROC'] = TechnicalIndicators.calculate_roc(self.df['close'], roc_period)
        self.indicators['ROC'] = True
        
        # Volume SMA
        self.df['Volume_SMA'] = TechnicalIndicators.calculate_volume_sma(self.df['volume'], volume_period)
        self.indicators['Volume_SMA'] = True
        
        # VWAP
        self.df['VWAP'] = TechnicalIndicators.calculate_vwap(
            self.df['high'],
            self.df['low'],
            self.df['close'],
            self.df['volume']
        )
        self.indicators['VWAP'] = True
        
        # ADX
        self.df['ADX'] = TechnicalIndicators.calculate_adx(
            self.df['high'],
            self.df['low'],
            self.df['close']
        )
        self.indicators['ADX'] = True
        
        return self.df
    
    def get_indicators_summary(self) -> Dict:
        """取得指標計算統計"""
        return {
            "total_indicators": len(self.indicators),
            "calculated_indicators": list(self.indicators.keys()),
            "data_shape": self.df.shape
        }
