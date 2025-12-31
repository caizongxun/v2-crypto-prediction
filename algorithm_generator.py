#!/usr/bin/env python3
"""
算法生成器 - 逆向工程三个指標

通过数据技巧逆向工程会出最优指標算法：
1. 趨勢指標 (Trend Indicator)
2. 方向指標 (Direction Indicator)
3. 波幅性指標 (Volatility Indicator)
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class IndicatorType(Enum):
    """指標类型"""
    TREND = "trend"
    DIRECTION = "direction"
    VOLATILITY = "volatility"


@dataclass
class AlgorithmConfig:
    """
    算法配置
    """
    lookback_period: int = 50  # 回治期閻
    threshold_buy: float = 0.6  # 买入信態閾值
    threshold_sell: float = 0.4  # 卖出信態閾值
    min_samples: int = 20  # 最少样本数


class TrendIndicatorGenerator:
    """
    趨勢指標生成器
    
    原理: 逆向分析上涨下淌筶向
    - 计算 EMA 的深度和弥散度
    - 使用 ATR 正规化
    - 结合 ADX 强度
    """
    
    def __init__(self, config: AlgorithmConfig = None):
        self.config = config or AlgorithmConfig()
        self.indicators = {}
    
    def generate(self, df: pd.DataFrame) -> Dict:
        """
        生成趨勢指標
        
        Args:
            df: OHLCV 数据
        
        Returns:
            Dict: 指標配置与算法
        """
        print("\n" + "="*70)
        print("[趨勢指標] 逆向工程")
        print("="*70)
        
        # Step 1: 计算 EMA 的最优參数
        print("\n[Step 1] 优化 EMA 參数...")
        ema_periods = self._optimize_ema_periods(df)
        print(f"  快速 EMA: {ema_periods['fast']} 根")
        print(f"  缓慢 EMA: {ema_periods['slow']} 根")
        
        # Step 2: 逆向 ATR 參数
        print("\n[Step 2] 优化 ATR 參数...")
        atr_period = self._optimize_atr_period(df)
        print(f"  ATR 期閻: {atr_period} 根")
        
        # Step 3: 构建趨勢指標公式
        print("\n[Step 3] 构建趨勢公式...")
        formula = self._build_trend_formula(ema_periods, atr_period)
        
        # Step 4: 验证算法
        print("\n[Step 4] 验证算法...")
        trend_values = self._calculate_trend(df, ema_periods, atr_period)
        print(f"  计算成功: {len(trend_values)} 个值")
        print(f"  值毁域: [{trend_values.min():.4f}, {trend_values.max():.4f}]")
        
        result = {
            "indicator_type": IndicatorType.TREND.value,
            "algorithm": formula,
            "parameters": {
                "ema_fast_period": ema_periods['fast'],
                "ema_slow_period": ema_periods['slow'],
                "atr_period": atr_period
            },
            "validation": {
                "total_values": len(trend_values),
                "min_value": float(trend_values.min()),
                "max_value": float(trend_values.max()),
                "mean_value": float(trend_values.mean()),
                "std_value": float(trend_values.std())
            }
        }
        
        return result
    
    def _optimize_ema_periods(self, df: pd.DataFrame) -> Dict:
        """
        通过逆向分析优化 EMA 參数
        """
        close = df['close']
        best_score = float('-inf')
        best_params = {'fast': 10, 'slow': 30}
        
        # 桂取上升趨勢段
        for fast in range(5, 25, 2):
            for slow in range(30, 100, 10):
                if fast >= slow:
                    continue
                
                ema_fast = close.ewm(span=fast, adjust=False).mean()
                ema_slow = close.ewm(span=slow, adjust=False).mean()
                
                # 计算梯度 (EMA 交是)
                crosses = ((ema_fast > ema_slow) != (ema_fast.shift(1) > ema_slow.shift(1))).sum()
                
                # 优化目标: 交待点数量适中
                score = -abs(crosses - 30)  # 理想是 20-40 个交待
                
                if score > best_score:
                    best_score = score
                    best_params = {'fast': fast, 'slow': slow}
        
        return best_params
    
    def _optimize_atr_period(self, df: pd.DataFrame) -> int:
        """
        通过逆向分析优化 ATR 參数
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        best_score = float('-inf')
        best_period = 14
        
        for period in range(10, 30, 2):
            # 计算 ATR
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.ewm(span=period, adjust=False).mean()
            
            # 优化目标: ATR 波幅功率控制
            atr_pct = (atr / close) * 100
            score = -abs(atr_pct.mean() - 0.5)  # 理想是 0.5% 上下
            
            if score > best_score:
                best_score = score
                best_period = period
        
        return best_period
    
    def _build_trend_formula(self, ema_periods: Dict, atr_period: int) -> str:
        """
        构建趨勢公式
        """
        formula = f"""
TREND INDICATOR FORMULA
======================

1. 计算优化后的 EMA:
   - fast_ema = EMA(close, {ema_periods['fast']})
   - slow_ema = EMA(close, {ema_periods['slow']})

2. 计算 ATR:
   - tr = max(high-low, abs(high-close[t-1]), abs(low-close[t-1]))
   - atr = EMA(tr, {atr_period})

3. 趨勢指標 (Trend Score):
   - ema_ratio = (fast_ema - slow_ema) / slow_ema * 100
   - atr_ratio = (atr / close) * 100
   - trend_score = tanh(ema_ratio / 2) * (1 - exp(-atr_ratio / 0.5))

4. 规格化为 [0, 1]:
   - trend_value = (trend_score + 1) / 2

使用:
   - trend_value > 0.6: 强上涨趨勢
   - trend_value > 0.5: 上涨趨勢
   - trend_value < 0.4: 下跌趨勢
   - trend_value < 0.3: 强下跌趨勢
"""
        return formula
    
    def _calculate_trend(self, df: pd.DataFrame, ema_periods: Dict, atr_period: int) -> pd.Series:
        """
        计算趨勢下标
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # EMA
        fast_ema = close.ewm(span=ema_periods['fast'], adjust=False).mean()
        slow_ema = close.ewm(span=ema_periods['slow'], adjust=False).mean()
        
        # ATR
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_period, adjust=False).mean()
        
        # 趨勢下标
        ema_ratio = (fast_ema - slow_ema) / slow_ema * 100
        atr_ratio = (atr / close) * 100
        
        trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-atr_ratio / 0.5))
        trend_value = (trend_score + 1) / 2
        
        return trend_value.clip(0, 1)


class DirectionIndicatorGenerator:
    """
    方向指標生成器
    
    原理: 逆向分析上涨下淌筶向
    - 计算上涨下淌段核
    - 使用动量指標 (RSI)
    - 结合成交量信息
    """
    
    def __init__(self, config: AlgorithmConfig = None):
        self.config = config or AlgorithmConfig()
    
    def generate(self, df: pd.DataFrame) -> Dict:
        """
        生成方向指標
        """
        print("\n" + "="*70)
        print("[方向指標] 逆向工程")
        print("="*70)
        
        # Step 1: 优化 RSI 參数
        print("\n[Step 1] 优化 RSI 參数...")
        rsi_period = self._optimize_rsi_period(df)
        print(f"  RSI 期閻: {rsi_period} 根")
        
        # Step 2: 优化 ROC 參数
        print("\n[Step 2] 优化 ROC 參数...")
        roc_period = self._optimize_roc_period(df)
        print(f"  ROC 期閻: {roc_period} 根")
        
        # Step 3: 构建方向公式
        print("\n[Step 3] 构建方向公式...")
        formula = self._build_direction_formula(rsi_period, roc_period)
        
        # Step 4: 验证
        print("\n[Step 4] 验证算法...")
        direction_values = self._calculate_direction(df, rsi_period, roc_period)
        direction_values = direction_values.dropna()
        
        print(f"  计算成功: {len(direction_values)} 个值")
        if len(direction_values) > 0:
            print(f"  值毁域: [{direction_values.min():.4f}, {direction_values.max():.4f}]")
        
        result = {
            "indicator_type": IndicatorType.DIRECTION.value,
            "algorithm": formula,
            "parameters": {
                "rsi_period": rsi_period,
                "roc_period": roc_period
            },
            "validation": {
                "total_values": len(direction_values),
                "min_value": float(direction_values.min()) if len(direction_values) > 0 else 0,
                "max_value": float(direction_values.max()) if len(direction_values) > 0 else 0,
                "mean_value": float(direction_values.mean()) if len(direction_values) > 0 else 0,
                "std_value": float(direction_values.std()) if len(direction_values) > 0 else 0
            }
        }
        
        return result
    
    def _optimize_rsi_period(self, df: pd.DataFrame) -> int:
        """
        优化 RSI 參数
        """
        close = df['close']
        best_score = float('-inf')
        best_period = 14
        
        for period in range(10, 25, 1):
            delta = close.diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean()
            avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # 优化目标: 超买超卖区域有效
            oversold = (rsi < 30).sum()
            overbought = (rsi > 70).sum()
            score = oversold + overbought
            
            if score > best_score:
                best_score = score
                best_period = period
        
        return best_period
    
    def _optimize_roc_period(self, df: pd.DataFrame) -> int:
        """
        优化 ROC 參数
        """
        close = df['close']
        best_score = float('-inf')
        best_period = 12
        
        for period in range(8, 20, 1):
            roc = ((close - close.shift(period)) / close.shift(period)) * 100
            
            # 优化目标: 正负值均衡
            score = -abs(roc.mean())
            
            if score > best_score:
                best_score = score
                best_period = period
        
        return best_period
    
    def _build_direction_formula(self, rsi_period: int, roc_period: int) -> str:
        """
        构建方向公式
        """
        formula = f"""
DIRECTION INDICATOR FORMULA
===========================

1. 计算 RSI:
   - delta = close[t] - close[t-1]
   - avg_gain = EMA(gain, {rsi_period})
   - avg_loss = EMA(loss, {rsi_period})
   - rs = avg_gain / avg_loss
   - rsi = 100 - (100 / (1 + rs))

2. 计算 ROC:
   - roc = ((close - close[t-{roc_period}]) / close[t-{roc_period}]) * 100

3. 方向指標 (Direction Score):
   - rsi_signal = (rsi - 50) / 50  # 规格化为 [-1, 1]
   - roc_signal = tanh(roc / 5)    # 规格化为 [-1, 1]
   - direction_score = (rsi_signal + roc_signal) / 2

4. 规格化为 [0, 1]:
   - direction_value = (direction_score + 1) / 2

使用:
   - direction_value > 0.6: 强上涨方向
   - direction_value > 0.5: 上涨方向
   - direction_value < 0.4: 下跌方向
   - direction_value < 0.3: 强下跌方向
"""
        return formula
    
    def _calculate_direction(self, df: pd.DataFrame, rsi_period: int, roc_period: int) -> pd.Series:
        """
        计算方向下标
        """
        close = df['close']
        
        # RSI
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # ROC
        roc = ((close - close.shift(roc_period)) / close.shift(roc_period)) * 100
        
        # 方向下标
        rsi_signal = (rsi - 50) / 50
        roc_signal = np.tanh(roc / 5)
        direction_score = (rsi_signal + roc_signal) / 2
        direction_value = (direction_score + 1) / 2
        
        return direction_value.clip(0, 1)


class VolatilityIndicatorGenerator:
    """
    波幅性指標生成器
    
    原理: 逆向分析价格波动
    - 计算标准差
    - 使用 Bollinger Bands
    - 结合波幅碩
    """
    
    def __init__(self, config: AlgorithmConfig = None):
        self.config = config or AlgorithmConfig()
    
    def generate(self, df: pd.DataFrame) -> Dict:
        """
        生成波幅性指標
        """
        print("\n" + "="*70)
        print("[波幅性指標] 逆向工程")
        print("="*70)
        
        # Step 1: 优化 SMA 參数
        print("\n[Step 1] 优化 SMA 參数...")
        sma_period = self._optimize_sma_period(df)
        print(f"  SMA 期閻: {sma_period} 根")
        
        # Step 2: 优化 BB 攜数
        print("\n[Step 2] 优化 Bollinger Bands 攜数...")
        bb_std = self._optimize_bb_std(df, sma_period)
        print(f"  BB 标准差倍数: {bb_std:.2f}")
        
        # Step 3: 构建波幅公式
        print("\n[Step 3] 构建波幅公式...")
        formula = self._build_volatility_formula(sma_period, bb_std)
        
        # Step 4: 验证
        print("\n[Step 4] 验证算法...")
        volatility_values = self._calculate_volatility(df, sma_period, bb_std)
        print(f"  计算成功: {len(volatility_values)} 个值")
        print(f"  值毁域: [{volatility_values.min():.4f}, {volatility_values.max():.4f}]")
        
        result = {
            "indicator_type": IndicatorType.VOLATILITY.value,
            "algorithm": formula,
            "parameters": {
                "sma_period": sma_period,
                "bb_std_multiplier": bb_std
            },
            "validation": {
                "total_values": len(volatility_values),
                "min_value": float(volatility_values.min()),
                "max_value": float(volatility_values.max()),
                "mean_value": float(volatility_values.mean()),
                "std_value": float(volatility_values.std())
            }
        }
        
        return result
    
    def _optimize_sma_period(self, df: pd.DataFrame) -> int:
        """
        优化 SMA 參数
        """
        close = df['close']
        best_score = float('-inf')
        best_period = 20
        
        for period in range(15, 40, 2):
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            
            # 优化目标: 标准差有效性
            score = std.std()
            
            if score > best_score:
                best_score = score
                best_period = period
        
        return best_period
    
    def _optimize_bb_std(self, df: pd.DataFrame, sma_period: int) -> float:
        """
        优化 Bollinger Bands 攜数
        """
        close = df['close']
        sma = close.rolling(window=sma_period).mean()
        std = close.rolling(window=sma_period).std()
        
        best_score = float('-inf')
        best_std = 2.0
        
        for bb_std in np.arange(1.5, 3.0, 0.1):
            upper = sma + (std * bb_std)
            lower = sma - (std * bb_std)
            
            # 优化目标: 价格突突条数
            breaks = ((close > upper) | (close < lower)).sum()
            score = -abs(breaks / len(df) - 0.05)  # 理想是 5% 突突
            
            if score > best_score:
                best_score = score
                best_std = bb_std
        
        return round(best_std, 2)
    
    def _build_volatility_formula(self, sma_period: int, bb_std: float) -> str:
        """
        构建波幅公式
        """
        formula = f"""
VOLATILITY INDICATOR FORMULA
============================

1. 计算 SMA 和 标准差:
   - sma = SMA(close, {sma_period})
   - std = STDEV(close, {sma_period})

2. 计算 Bollinger Bands:
   - upper = sma + ({bb_std} * std)
   - lower = sma - ({bb_std} * std)
   - bb_width = (upper - lower) / sma

3. 波幅性指標 (Volatility Score):
   - bandwidth = (close - lower) / (upper - lower)  # [0, 1]
   - volatility_ratio = std / sma * 100
   - volatility_score = sqrt(volatility_ratio / 2)

4. 最终下标 [0, 1]:
   - volatility_value = min(volatility_score, 1.0)

使用:
   - volatility_value > 0.6: 高波幅性 (足以下上洋)
   - volatility_value > 0.4: 中低波幅性
   - volatility_value < 0.3: 低波幅性 (波潫整理)
"""
        return formula
    
    def _calculate_volatility(self, df: pd.DataFrame, sma_period: int, bb_std: float) -> pd.Series:
        """
        计算波幅性下标
        """
        close = df['close']
        
        # SMA 和 标准差
        sma = close.rolling(window=sma_period).mean()
        std = close.rolling(window=sma_period).std()
        
        # Bollinger Bands
        upper = sma + (std * bb_std)
        lower = sma - (std * bb_std)
        
        # 波幅性下标
        volatility_ratio = std / sma * 100
        volatility_score = np.sqrt(volatility_ratio / 2)
        volatility_value = volatility_score.clip(0, 1)
        
        return volatility_value


def main():
    """
    主流程
    """
    print("\n" + "#"*70)
    print("# 算法生成器 - 三个指標的逆向工程")
    print("#"*70)
    
    # 加載数据
    print("\n[一] 加載数据...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        
        # 過濾日期
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        print(f"\n数据加載成功")
        print(f"时閻范围: {df.index[0]} ~ {df.index[-1]}")
        print(f"根数: {len(df)}")
    except Exception as e:
        print(f"数据加載失败: {e}")
        print(f"\n请先运行: python download_and_save_data.py")
        return
    
    # 生成三个指標
    config = AlgorithmConfig()
    
    results = {}
    
    # 1. 趨勢指標
    generator1 = TrendIndicatorGenerator(config)
    results['trend'] = generator1.generate(df)
    
    # 2. 方向指標
    generator2 = DirectionIndicatorGenerator(config)
    results['direction'] = generator2.generate(df)
    
    # 3. 波幅性指標
    generator3 = VolatilityIndicatorGenerator(config)
    results['volatility'] = generator3.generate(df)
    
    # 保存结果
    print("\n" + "#"*70)
    print("# 保存结果")
    print("#"*70)
    
    os.makedirs("results", exist_ok=True)
    
    output_file = "results/algorithm_generation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存: {output_file}")
    
    # 打印汇总
    print("\n" + "="*70)
    print("汇总")
    print("="*70)
    
    for name, result in results.items():
        print(f"\n{name.upper()} INDICATOR:")
        print(f"  类型: {result['indicator_type']}")
        print(f"  參数: {result['parameters']}")
        print(f"  验证: {len(result['validation'])} 个数据")
    
    print("\n" + "#"*70)
    print("# 下一步: python backtest_indicators.py")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
