"""
黃金公式 V2 - 配置結構

理論結構：
1. 趥勢 - 多時間框架 EMA
2. 波動率 - ATR 篩選
3. 動能 - RSI / Stoch RSI / ROC 組合
4. 成交量 - 成交量高峰 / VWAP 偏離
"""

from dataclasses import dataclass
from typing import Dict, Tuple
from enum import Enum


class TrendComponent(Enum):
    """趥勢組件"""
    MULTI_TIMEFRAME_EMA = "multi_timeframe_ema"
    SUPERTREND = "supertrend"
    ADX = "adx"


class VolatilityComponent(Enum):
    """波動率組件"""
    ATR = "atr"
    ATR_RATIO = "atr_ratio"


class MomentumComponent(Enum):
    """動能組件"""
    RSI = "rsi"
    STOCH_RSI = "stoch_rsi"
    ROC = "roc"


class VolumeComponent(Enum):
    """成交量組件"""
    VOLUME_SPIKE = "volume_spike"
    VWAP_DEVIATION = "vwap_deviation"


@dataclass
class TrendConfig:
    """
    趥勢配置
    
    多時間框架 EMA 方向一致：
    - EMA_15m: 15時間 EMA (缥德的子)
    - EMA_1h: 60時間 EMA (父的趥勢)
    """
    
    # 時間間隔 (单位: 根 K 線)
    fast_ema_period: int = 15    # 15 min 時間框架 = 15根 K 線
    slow_ema_period: int = 60    # 1 hour 時間框架 = 60根 K 線
    
    # SuperTrend 參數
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    
    # ADX 參數
    adx_period: int = 14
    adx_min_threshold: float = 25.0  # ADX 最低似置：趥勢強度
    
    # 趥勢方向定義
    bullish_config: Dict = None  # EMA_15 > EMA_60, SuperTrend 幻趥, ADX > 閉值
    bearish_config: Dict = None  # EMA_15 < EMA_60, SuperTrend 窗趥, ADX > 閉值
    
    def __post_init__(self):
        if self.bullish_config is None:
            self.bullish_config = {
                "name": "Bullish",
                "ema_condition": "fast > slow",
                "supertrend_condition": "uptrend",
                "signal": "BUY"
            }
        
        if self.bearish_config is None:
            self.bearish_config = {
                "name": "Bearish",
                "ema_condition": "fast < slow",
                "supertrend_condition": "downtrend",
                "signal": "SELL"
            }


@dataclass
class VolatilityConfig:
    """
    波動率配置
    
    篩選統核：
    - ATR 太低: 波動率不足，不作业
    - ATR 太高: 可能是崂挺，需要警惑
    """
    
    atr_period: int = 14
    
    # ATR 篩選浜低閉值
    min_atr_percent: float = 0.5  # ATR 低於收盤價的 0.5% 時不作业
    
    # ATR 警惑上限 (可選)
    max_atr_percent: float = 2.0  # ATR 高於收盤價的 2% 時需警惑


@dataclass
class MomentumConfig:
    """
    動能配置
    
    RSI / Stoch RSI / ROC 組合判斷底頂和頂部
    """
    
    # RSI 參數
    rsi_period: int = 14
    rsi_oversold: float = 30  # 超購区间 (low momentum)
    rsi_overbought: float = 70  # 超賣区间 (high momentum)
    
    # Stochastic RSI 參數
    stoch_rsi_period: int = 14
    stoch_rsi_smooth_k: int = 3
    stoch_rsi_smooth_d: int = 3
    stoch_rsi_oversold: float = 20
    stoch_rsi_overbought: float = 80
    
    # ROC 參數
    roc_period: int = 12
    roc_positive_threshold: float = 0.0  # ROC > 0 = 上漲動能
    
    # 動能策略
    buy_condition: str = "RSI > 50 AND Stoch_RSI_K > Stoch_RSI_D AND ROC > 0"
    sell_condition: str = "RSI < 50 AND Stoch_RSI_K < Stoch_RSI_D AND ROC < 0"


@dataclass
class VolumeConfig:
    """
    成交量配置
    
    成交量高峰 和 VWAP 偏離
    """
    
    # 成交量 SMA
    volume_sma_period: int = 20
    volume_spike_multiplier: float = 1.5  # 成交量 > SMA * 1.5 為高峰
    
    # VWAP 參數
    vwap_period: int = 20
    vwap_deviation_percent: float = 1.0  # 價格偏離 VWAP > 1.0% 為不尋常


@dataclass
class EntryConfig:
    """
    出場配置
    
    綜合所有指標的出場條件
    """
    
    # 權重 (需要采集)
    trend_weight: float = 0.4  # 趥勢: 40%
    momentum_weight: float = 0.3  # 動能: 30%
    volume_weight: float = 0.2  # 成交量: 20%
    volatility_weight: float = 0.1  # 波動率: 10% (作為篩選）
    
    # 總体信忆稿閉值 (最終作業/不作業的墊樣)
    min_confidence_threshold: float = 0.65  # 信忆稿 >= 65% 膬出場
    
    # 宗緒控制
    max_consecutive_signals: int = 5  # 不超過 5 个連續信號


@dataclass
class GoldenFormulaV2Config:
    """
    黃金公式 V2 完整配置
    
    理論結構：
    Trend (40%) + Momentum (30%) + Volume (20%) + Volatility Filter (10%)
    """
    
    trend_config: TrendConfig = None
    volatility_config: VolatilityConfig = None
    momentum_config: MomentumConfig = None
    volume_config: VolumeConfig = None
    entry_config: EntryConfig = None
    
    def __post_init__(self):
        if self.trend_config is None:
            self.trend_config = TrendConfig()
        if self.volatility_config is None:
            self.volatility_config = VolatilityConfig()
        if self.momentum_config is None:
            self.momentum_config = MomentumConfig()
        if self.volume_config is None:
            self.volume_config = VolumeConfig()
        if self.entry_config is None:
            self.entry_config = EntryConfig()
    
    def get_config_summary(self) -> Dict:
        """取得配置摆要"""
        return {
            "trend": {
                "fast_ema": self.trend_config.fast_ema_period,
                "slow_ema": self.trend_config.slow_ema_period,
                "supertrend_period": self.trend_config.supertrend_period,
                "supertrend_multiplier": self.trend_config.supertrend_multiplier,
                "adx_min_threshold": self.trend_config.adx_min_threshold,
            },
            "volatility": {
                "atr_period": self.volatility_config.atr_period,
                "min_atr_percent": self.volatility_config.min_atr_percent,
                "max_atr_percent": self.volatility_config.max_atr_percent,
            },
            "momentum": {
                "rsi_period": self.momentum_config.rsi_period,
                "rsi_oversold": self.momentum_config.rsi_oversold,
                "rsi_overbought": self.momentum_config.rsi_overbought,
                "stoch_rsi_oversold": self.momentum_config.stoch_rsi_oversold,
                "stoch_rsi_overbought": self.momentum_config.stoch_rsi_overbought,
                "roc_period": self.momentum_config.roc_period,
            },
            "volume": {
                "volume_sma_period": self.volume_config.volume_sma_period,
                "volume_spike_multiplier": self.volume_config.volume_spike_multiplier,
                "vwap_deviation_percent": self.volume_config.vwap_deviation_percent,
            },
            "entry": {
                "trend_weight": self.entry_config.trend_weight,
                "momentum_weight": self.entry_config.momentum_weight,
                "volume_weight": self.entry_config.volume_weight,
                "volatility_weight": self.entry_config.volatility_weight,
                "min_confidence_threshold": self.entry_config.min_confidence_threshold,
            }
        }
    
    def print_config(self):
        """列印配置"""
        print("\n" + "=" * 70)
        print("黃金公式 V2 配置")
        print("=" * 70)
        
        summary = self.get_config_summary()
        
        for section, params in summary.items():
            print(f"\n{section.upper()} 組件:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 70)
        print(f"權重: Trend {self.entry_config.trend_weight*100:.0f}% + "
              f"Momentum {self.entry_config.momentum_weight*100:.0f}% + "
              f"Volume {self.entry_config.volume_weight*100:.0f}% + "
              f"Volatility {self.entry_config.volatility_weight*100:.0f}%")
        print(f"最低信忆稿閉值: {self.entry_config.min_confidence_threshold*100:.0f}%")
        print("=" * 70 + "\n")
