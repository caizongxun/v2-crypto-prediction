"""
黃金公式 V1 - 區間反轉樣式偵測

遮鑊宰陰陳明帏 (或布林之罂) 極限樣式樣子的極限值
當一个区间的视角是黑色时，接下来的K线会是白色，体现为上帆斧下普
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """信号符号"""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


@dataclass
class Pattern:
    """樣式结构"""
    timestamp: pd.Timestamp
    index: int
    signal: Signal
    confidence: float
    pattern_type: str
    details: Dict


class GoldenFormulaV1:
    """
    黃金公式 V1 - 區間反轉樣式偵測
    """
    
    def __init__(self, lookback_period: int = 20):
        """
        初始化公式
        
        Args:
            lookback_period: 回的视野页数 (K线数)
        """
        self.lookback_period = lookback_period
        self.patterns = []
        
    def detect_interval_reversal(
        self,
        df: pd.DataFrame,
        min_pattern_strength: float = 0.7
    ) -> List[Pattern]:
        """
        检测区间反轉樣式
        
        黑色K线特征: close < open
        白色K线特征: close > open
        
        Args:
            df: OHLCV 数据
            min_pattern_strength: 最低樣式强度
        
        Returns:
            List[Pattern]: 检测到的樣式
        """
        patterns = []
        
        if len(df) < self.lookback_period + 2:
            return patterns
        
        # 标记K线颜色 (1=白色 close>open, -1=黑色 close<open, 0=同佋线)
        df_work = df.copy()
        df_work['color'] = np.where(
            df_work['close'] > df_work['open'],
            1,
            np.where(df_work['close'] < df_work['open'], -1, 0)
        )
        
        # 检测樣式
        for i in range(self.lookback_period, len(df_work) - 1):
            # 当前 K线是黑色 (反轉信号)
            if df_work.iloc[i]['color'] == -1:
                # 检查之前是否有白色K线
                prev_lookback = df_work.iloc[i-self.lookback_period:i]
                
                if (prev_lookback['color'] == 1).any():
                    # 下一根 K线会是白色
                    next_candle = df_work.iloc[i+1]
                    
                    if next_candle['color'] == 1:
                        # 计算信忆稿
                        confidence = self._calculate_confidence(
                            df_work.iloc[i-self.lookback_period:i+2]
                        )
                        
                        if confidence >= min_pattern_strength:
                            pattern = Pattern(
                                timestamp=df.iloc[i].name if isinstance(df.index[0], pd.Timestamp) else df.iloc[i]['open_time'],
                                index=i,
                                signal=Signal.BUY,
                                confidence=confidence,
                                pattern_type="interval_reversal",
                                details={
                                    "open": float(df.iloc[i]['open']),
                                    "high": float(df.iloc[i]['high']),
                                    "low": float(df.iloc[i]['low']),
                                    "close": float(df.iloc[i]['close']),
                                    "lookback_period": self.lookback_period,
                                }
                            )
                            patterns.append(pattern)
            
            # 当前 K线是白色 (反轉信号)
            elif df_work.iloc[i]['color'] == 1:
                prev_lookback = df_work.iloc[i-self.lookback_period:i]
                
                if (prev_lookback['color'] == -1).any():
                    next_candle = df_work.iloc[i+1]
                    
                    if next_candle['color'] == -1:
                        confidence = self._calculate_confidence(
                            df_work.iloc[i-self.lookback_period:i+2]
                        )
                        
                        if confidence >= min_pattern_strength:
                            pattern = Pattern(
                                timestamp=df.iloc[i].name if isinstance(df.index[0], pd.Timestamp) else df.iloc[i]['open_time'],
                                index=i,
                                signal=Signal.SELL,
                                confidence=confidence,
                                pattern_type="interval_reversal",
                                details={
                                    "open": float(df.iloc[i]['open']),
                                    "high": float(df.iloc[i]['high']),
                                    "low": float(df.iloc[i]['low']),
                                    "close": float(df.iloc[i]['close']),
                                    "lookback_period": self.lookback_period,
                                }
                            )
                            patterns.append(pattern)
        
        self.patterns = patterns
        return patterns
    
    def _calculate_confidence(self, lookback_df: pd.DataFrame) -> float:
        """
        计算樣式强度/信忆稿
        
        基于:
        1. 樣式的清晰度
        2. 体量模式
        3. 价樋樣式
        
        Args:
            lookback_df: 回的数据
        
        Returns:
            float: 信忆稿 (0-1)
        """
        scores = []
        
        # 1. K线颜色一致性
        colors = np.where(
            lookback_df['close'] > lookback_df['open'],
            1,
            np.where(lookback_df['close'] < lookback_df['open'], -1, 0)
        )
        
        # 统计颜色一致性
        if len(colors) > 0:
            unique_colors = len(set(colors[colors != 0]))
            color_consistency = 1.0 - (unique_colors - 1) / 2
            scores.append(color_consistency * 0.3)
        
        # 2. 体量强动
        if 'volume' in lookback_df.columns:
            avg_volume = lookback_df['volume'].mean()
            current_volume = lookback_df['volume'].iloc[-1]
            volume_ratio = min(current_volume / avg_volume if avg_volume > 0 else 1, 2.0)
            scores.append(min(volume_ratio / 2, 1.0) * 0.3)
        
        # 3. 价格范围
        high_low_range = (lookback_df['high'] - lookback_df['low']).mean()
        if high_low_range > 0:
            current_range = lookback_df['high'].iloc[-1] - lookback_df['low'].iloc[-1]
            range_ratio = current_range / high_low_range
            scores.append(min(range_ratio / 1.5, 1.0) * 0.4)
        
        confidence = sum(scores) if scores else 0.5
        return min(max(confidence, 0.0), 1.0)
    
    def get_last_pattern(self) -> Optional[Pattern]:
        """取得最后一个检测到的樣式"""
        return self.patterns[-1] if self.patterns else None
    
    def get_patterns_summary(self) -> Dict:
        """获取樣式汇总"""
        if not self.patterns:
            return {"total": 0, "buy_signals": 0, "sell_signals": 0}
        
        buy_count = sum(1 for p in self.patterns if p.signal == Signal.BUY)
        sell_count = sum(1 for p in self.patterns if p.signal == Signal.SELL)
        
        return {
            "total": len(self.patterns),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "average_confidence": np.mean([p.confidence for p in self.patterns])
        }
