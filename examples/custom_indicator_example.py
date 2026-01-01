"""
自定義指標示例
此檔案演示如何把轉換後的 Pine Script 代碼
轉變成一個完成的指標類。
"""

import pandas as pd
import numpy as np
from sys import path
path.append('..')

from indicator_framework import BaseIndicator, IndicatorManager


class AdvancedRSIIndicator(BaseIndicator):
    """
    轉換自 Pine Script 的進阱 RSI 指標
    支援可調整的參數和處理信號
    """
    
    def __init__(self):
        super().__init__(
            name='Advanced RSI',
            description='Enhanced RSI with multiple signal detection',
            overlay=False
        )
        
        # 一點候提鈴流
        self.parameters = {
            'period': 14,
            'overbought': 70,
            'oversold': 30,
            'smooth_period': 3,
        }
    
    def calculate(self, df: pd.DataFrame):
        """計算 Advanced RSI 指標"""
        close = df['close']
        
        # 計算基癀 RSI
        rsi = self._calculate_rsi(close, self.parameters['period'])
        
        # 平滑 RSI (可選)
        if self.parameters.get('smooth_period', 1) > 1:
            rsi_smooth = rsi.rolling(window=self.parameters['smooth_period']).mean()
        else:
            rsi_smooth = rsi
        
        # 中中線
        rsi_sma = rsi.rolling(window=50).mean()
        
        # 添加主線
        self.add_plot('RSI', rsi, color='#1E90FF', width=2.0)
        self.add_plot('RSI Smoothed', rsi_smooth, color='#FFD700', width=1.5, alpha=0.8)
        
        # 添加基準線
        self.add_plot(
            'Overbought',
            pd.Series([self.parameters['overbought']] * len(rsi), index=rsi.index),
            color='#FF0000', line_style='--', width=1.0, alpha=0.6
        )
        self.add_plot(
            'Oversold',
            pd.Series([self.parameters['oversold']] * len(rsi), index=rsi.index),
            color='#00FF00', line_style='--', width=1.0, alpha=0.6
        )
        self.add_plot(
            'Middle',
            pd.Series([50] * len(rsi), index=rsi.index),
            color='#CCCCCC', line_style=':', width=1.0, alpha=0.5
        )
        
        # 生成信號
        self._generate_signals(rsi, rsi_smooth)
        
        return {
            'plots': self.plots,
            'signals': self.signals,
        }
    
    def _calculate_rsi(self, data, period):
        """計算 RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _generate_signals(self, rsi, rsi_smooth):
        """生成交易信號"""
        for i in range(1, len(rsi)):
            if pd.isna(rsi.iloc[i]):
                continue
            
            rsi_val = rsi.iloc[i]
            rsi_prev = rsi.iloc[i-1]
            
            # 超資休增強
            if rsi_prev <= self.parameters['overbought'] and rsi_val > self.parameters['overbought']:
                self.add_signal(
                    'overbought_entry',
                    i,
                    rsi_val,
                    {'signal_strength': rsi_val - self.parameters['overbought']}
                )
            
            # 超資休出場
            if rsi_prev > self.parameters['overbought'] and rsi_val <= self.parameters['overbought']:
                self.add_signal(
                    'overbought_exit',
                    i,
                    rsi_val,
                    {'signal_strength': self.parameters['overbought'] - rsi_val}
                )
            
            # 超賣水位增強
            if rsi_prev >= self.parameters['oversold'] and rsi_val < self.parameters['oversold']:
                self.add_signal(
                    'oversold_entry',
                    i,
                    rsi_val,
                    {'signal_strength': self.parameters['oversold'] - rsi_val}
                )
            
            # 超賣水位出場
            if rsi_prev < self.parameters['oversold'] and rsi_val >= self.parameters['oversold']:
                self.add_signal(
                    'oversold_exit',
                    i,
                    rsi_val,
                    {'signal_strength': rsi_val - self.parameters['oversold']}
                )
            
            # 席位中線空頭 (divergence)
            if i > 50 and not pd.isna(rsi_smooth.iloc[i-50:i].mean()):
                rsi_trend = rsi.iloc[i] - rsi.iloc[i-50]
                if rsi_trend < -10:  # 下降趨勢明顏
                    self.add_signal(
                        'divergence_bearish',
                        i,
                        rsi_val,
                        {'trend': rsi_trend}
                    )
                elif rsi_trend > 10:  # 上滺趨勢
                    self.add_signal(
                        'divergence_bullish',
                        i,
                        rsi_val,
                        {'trend': rsi_trend}
                    )


class StochasticIndicator(BaseIndicator):
    """
    隊機指標
    """
    
    def __init__(self):
        super().__init__(
            name='Stochastic Oscillator',
            description='Stochastic Oscillator with K and D lines',
            overlay=False
        )
        
        self.parameters = {
            'k_period': 14,
            'd_period': 3,
            'overbought': 80,
            'oversold': 20,
        }
    
    def calculate(self, df: pd.DataFrame):
        """計算隊機指標"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 計算最高低價
        lowest_low = low.rolling(window=self.parameters['k_period']).min()
        highest_high = high.rolling(window=self.parameters['k_period']).max()
        
        # 計算 K 值
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # 計算 D 值 (信號線)
        d_percent = k_percent.rolling(window=self.parameters['d_period']).mean()
        
        # 添加線
        self.add_plot('K%', k_percent, color='#1E90FF', width=1.5)
        self.add_plot('D%', d_percent, color='#FF6347', width=1.5)
        
        # 添加參考線
        self.add_plot(
            'Overbought',
            pd.Series([self.parameters['overbought']] * len(k_percent), index=k_percent.index),
            color='#FF0000', line_style='--', width=1.0, alpha=0.6
        )
        self.add_plot(
            'Oversold',
            pd.Series([self.parameters['oversold']] * len(k_percent), index=k_percent.index),
            color='#00FF00', line_style='--', width=1.0, alpha=0.6
        )
        
        # 生成信號
        for i in range(1, len(k_percent)):
            if pd.isna(k_percent.iloc[i]) or pd.isna(d_percent.iloc[i]):
                continue
            
            k_val = k_percent.iloc[i]
            d_val = d_percent.iloc[i]
            k_prev = k_percent.iloc[i-1]
            d_prev = d_percent.iloc[i-1]
            
            # K 超逾 D - Bullish Signal
            if k_prev <= d_prev and k_val > d_val:
                self.add_signal('bullish_cross', i, k_val)
            
            # K 超邊 D - Bearish Signal
            if k_prev >= d_prev and k_val < d_val:
                self.add_signal('bearish_cross', i, k_val)
            
            # 超資休
            if k_val > self.parameters['overbought']:
                self.add_signal('overbought', i, k_val)
            
            # 超賣水位
            if k_val < self.parameters['oversold']:
                self.add_signal('oversold', i, k_val)
        
        return {
            'plots': self.plots,
            'signals': self.signals,
        }


def example_usage():
    """使用示例"""
    
    # 加載數據
    print("Loading data...")
    df = pd.read_parquet('../data/btc_15m.parquet')
    print(f"Loaded {len(df)} rows of data")
    
    # 創建指標管理器
    print("Creating indicator manager...")
    manager = IndicatorManager()
    
    # 註冊指標
    print("Registering indicators...")
    manager.register_indicator(AdvancedRSIIndicator())
    manager.register_indicator(StochasticIndicator())
    
    # 計算指標
    print("Calculating indicators...")
    results = manager.calculate_all(df)
    
    # 輸出信號
    print("\n=== Generated Signals ===")
    for name, indicator in manager.indicators.items():
        if indicator.signals:
            print(f"\n{name}:")
            for signal in indicator.signals[:10]:  # 首先 10 個信號
                print(f"  - {signal['type']} at index {signal['index']}: {signal['value']:.2f}")
    
    # 繪費
    print("\nPlotting indicators...")
    fig = manager.plot_indicators(df)
    
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    example_usage()
