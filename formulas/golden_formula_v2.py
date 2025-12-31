"""
黃金公式 V2 - 完整實現

綜合評分系統、加權綜合、評分邏輯
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from formulas.indicators import IndicatorCalculator
from formulas.golden_formula_v2_config import GoldenFormulaV2Config, TrendConfig, MomentumConfig, VolumeConfig, VolatilityConfig, EntryConfig


class Signal(Enum):
    """信號符号"""
    BUY = 1
    SELL = -1
    NEUTRAL = 0


@dataclass
class PatternV2:
    """樣式結模"""
    timestamp: pd.Timestamp
    index: int
    signal: Signal
    confidence: float
    
    # 組件分數
    trend_score: float
    momentum_score: float
    volume_score: float
    
    # 詳細數據
    details: Dict


class GoldenFormulaV2:
    """
    黃金公式 V2 - 完整實現
    """
    
    def __init__(self, config: GoldenFormulaV2Config = None):
        """
        初始化公式
        
        Args:
            config: 配置對象（預設使用預設配置）
        """
        self.config = config or GoldenFormulaV2Config()
        self.patterns = []
        self.df = None
        
    def analyze(self, df: pd.DataFrame) -> Tuple[List[PatternV2], pd.DataFrame]:
        """
        分析數據並產生信號
        
        Args:
            df: OHLCV 數據
        
        Returns:
            Tuple[List[PatternV2], pd.DataFrame]: (信號列表, 有指標的 dataframe)
        """
        self.df = df.copy()
        
        # 計算所有指標
        calc = IndicatorCalculator(self.df)
        self.df = calc.calculate_all_indicators(
            ema_periods=[self.config.trend_config.fast_ema_period, 
                        self.config.trend_config.slow_ema_period]
        )
        
        # 检測樣式
        patterns = []
        for i in range(self.config.trend_config.slow_ema_period, len(self.df) - 1):
            pattern = self._check_signal(i)
            if pattern:
                patterns.append(pattern)
        
        self.patterns = patterns
        return patterns, self.df
    
    def _check_signal(self, index: int) -> Optional[PatternV2]:
        """
        检查单一根 K 線是否有信號
        
        Args:
            index: K 線索引
        
        Returns:
            Optional[PatternV2]: 信號或 None
        """
        # 步驟 1: 波動率篩選
        if not self._volatility_filter(index):
            return None
        
        # 步驟 2: 計算各組件分數
        trend_score = self._calculate_trend_score(index)
        momentum_score = self._calculate_momentum_score(index)
        volume_score = self._calculate_volume_score(index)
        
        # 步驟 3: 加權綜合
        confidence = (
            trend_score * self.config.entry_config.trend_weight +
            momentum_score * self.config.entry_config.momentum_weight +
            volume_score * self.config.entry_config.volume_weight
        )
        
        # 步驟 4: 信新符算
        if confidence < self.config.entry_config.min_confidence_threshold:
            return None
        
        # 步驟 5: 確定信號方向
        row = self.df.iloc[index]
        signal = self._determine_signal(trend_score, momentum_score, volume_score)
        
        if signal == Signal.NEUTRAL:
            return None
        
        pattern = PatternV2(
            timestamp=row.name if isinstance(row.name, pd.Timestamp) else row.get('open_time', pd.Timestamp.now()),
            index=index,
            signal=signal,
            confidence=confidence,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            details={
                "close": float(row['close']),
                "high": float(row['high']),
                "low": float(row['low']),
                "volume": float(row.get('volume', 0)),
                "rsi": float(row.get('RSI', np.nan)),
                "atr": float(row.get('ATR', np.nan)),
                "ema_fast": float(row.get(f"EMA_{self.config.trend_config.fast_ema_period}", np.nan)),
                "ema_slow": float(row.get(f"EMA_{self.config.trend_config.slow_ema_period}", np.nan)),
            }
        )
        
        return pattern
    
    def _volatility_filter(self, index: int) -> bool:
        """
        波動率篩選 - 如果波動率不符合門檻則跳過
        
        Args:
            index: K 線索引
        
        Returns:
            bool: 是否通過篩選
        """
        row = self.df.iloc[index]
        atr = row.get('ATR', np.nan)
        close = row['close']
        
        if np.isnan(atr):
            return False
        
        atr_percent = (atr / close) * 100
        
        # ATR 太低: 過濾
        if atr_percent < self.config.volatility_config.min_atr_percent:
            return False
        
        # ATR 太高: 警告（可選）
        # return True 表示仍然接受，可改成 return False
        if atr_percent > self.config.volatility_config.max_atr_percent:
            return True  # 可以調整這一步
        
        return True
    
    def _calculate_trend_score(self, index: int) -> float:
        """
        計算趨勢分數 (40%)
        
        Args:
            index: K 線索引
        
        Returns:
            float: 0-1 之間的分數
        """
        row = self.df.iloc[index]
        
        fast_ema_col = f"EMA_{self.config.trend_config.fast_ema_period}"
        slow_ema_col = f"EMA_{self.config.trend_config.slow_ema_period}"
        
        fast_ema = row.get(fast_ema_col, np.nan)
        slow_ema = row.get(slow_ema_col, np.nan)
        supertrend = row.get('SuperTrend', np.nan)
        adx = row.get('ADX', np.nan)
        close = row['close']
        
        if any(np.isnan(x) for x in [fast_ema, slow_ema, supertrend, adx]):
            return 0.0
        
        # 1. EMA 方向
        ema_direction = 1 if fast_ema > slow_ema else (-1 if fast_ema < slow_ema else 0)
        
        # 2. SuperTrend 方向確認
        supertrend_direction = 1 if close > supertrend else (-1 if close < supertrend else 0)
        
        # 3. ADX 強度
        adx_strength = min(adx / 100, 1.0)
        
        # 4. 趨勢確認 (需要 EMA 和 SuperTrend 方向一致)
        if ema_direction == supertrend_direction and ema_direction != 0:
            trend_score = abs(ema_direction) * adx_strength
        else:
            trend_score = 0.0
        
        return min(max(trend_score, 0.0), 1.0)
    
    def _calculate_momentum_score(self, index: int) -> float:
        """
        計算動能分數 (30%)
        
        Args:
            index: K 線索引
        
        Returns:
            float: 0-1 之間的分數
        """
        row = self.df.iloc[index]
        
        rsi = row.get('RSI', np.nan)
        k_val = row.get('Stoch_RSI_K', np.nan)
        d_val = row.get('Stoch_RSI_D', np.nan)
        roc = row.get('ROC', np.nan)
        
        if any(np.isnan(x) for x in [rsi, k_val, d_val, roc]):
            return 0.0
        
        # 1. RSI 信號
        rsi_signal = 1 if rsi > 50 else (-1 if rsi < 50 else 0)
        
        # 2. Stoch RSI 確認
        stoch_signal = 1 if k_val > d_val else (-1 if k_val < d_val else 0)
        
        # 3. ROC 確認
        roc_signal = 1 if roc > 0 else (-1 if roc < 0 else 0)
        
        # 4. 投票系統 (需要至少 2/3 同意)
        signals = [rsi_signal, stoch_signal, roc_signal]
        momentum_votes = sum(1 for s in signals if s > 0)
        
        if momentum_votes >= 2:
            momentum_score = 1.0
        elif momentum_votes == 1:
            momentum_score = 0.5
        else:
            # 所有那些很下訊
            bear_votes = sum(1 for s in signals if s < 0)
            if bear_votes >= 2:
                momentum_score = 0.0
            else:
                momentum_score = 0.25
        
        return momentum_score
    
    def _calculate_volume_score(self, index: int) -> float:
        """
        計算成交量分數 (20%)
        
        Args:
            index: K 線索引
        
        Returns:
            float: 0-1 之間的分數
        """
        row = self.df.iloc[index]
        
        volume = row.get('volume', 0)
        volume_sma = row.get('Volume_SMA', np.nan)
        vwap = row.get('VWAP', np.nan)
        close = row['close']
        
        if np.isnan(volume_sma) or np.isnan(vwap):
            return 0.5
        
        # 1. 成交量高峰
        volume_spike = volume > (volume_sma * self.config.volume_config.volume_spike_multiplier)
        
        # 2. VWAP 偏離
        vwap_deviation_percent = (close - vwap) / vwap * 100
        vwap_bullish = vwap_deviation_percent > self.config.volume_config.vwap_deviation_percent
        vwap_bearish = vwap_deviation_percent < -self.config.volume_config.vwap_deviation_percent
        
        # 3. 綜合評分
        score = 0.5
        
        if volume_spike:
            score += 0.25
        
        if vwap_bullish:
            score += 0.25
        elif vwap_bearish:
            score -= 0.25
        
        return min(max(score, 0.0), 1.0)
    
    def _determine_signal(self, trend_score: float, momentum_score: float, volume_score: float) -> Signal:
        """
        根據各組件分數確定信號
        
        Args:
            trend_score: 趨勢分數
            momentum_score: 動能分數
            volume_score: 成交量分數
        
        Returns:
            Signal: 信號粗項
        """
        if trend_score > 0.5:
            return Signal.BUY
        elif trend_score < 0.5:
            return Signal.SELL
        else:
            return Signal.NEUTRAL
    
    def get_patterns_summary(self) -> Dict:
        """取得樣式汇总"""
        if not self.patterns:
            return {
                "total": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "average_confidence": 0.0,
                "win_rate": 0.0
            }
        
        buy_count = sum(1 for p in self.patterns if p.signal == Signal.BUY)
        sell_count = sum(1 for p in self.patterns if p.signal == Signal.SELL)
        avg_confidence = np.mean([p.confidence for p in self.patterns])
        
        return {
            "total": len(self.patterns),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "average_confidence": avg_confidence,
            "buy_ratio": buy_count / len(self.patterns) if self.patterns else 0.0
        }
    
    def get_last_pattern(self) -> Optional[PatternV2]:
        """取得最后一個樣式"""
        return self.patterns[-1] if self.patterns else None
