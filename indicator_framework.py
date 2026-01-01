import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import json


class BaseIndicator(ABC):
    """
    所有指標的基氵類
    """
    
    def __init__(self, name: str, description: str = '', overlay: bool = False):
        self.name = name
        self.description = description
        self.overlay = overlay  # True: 絢在K線上, False: 置於下方
        self.plots = {}  # 存放繪圖數沛
        self.signals = []  # 存放信歷
        self.parameters = {}  # 參數
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """計算指標,必須實袴"""
        pass
    
    def set_parameter(self, param_name: str, value: Any):
        """設置參數"""
        self.parameters[param_name] = value
    
    def add_plot(self, plot_name: str, data: pd.Series, color: str, 
                 line_style: str = '-', width: float = 1.5, alpha: float = 1.0):
        """添加繪圖數沛"""
        if plot_name not in self.plots:
            self.plots[plot_name] = []
        
        self.plots[plot_name].append({
            'data': data,
            'color': color,
            'line_style': line_style,
            'width': width,
            'alpha': alpha,
        })
    
    def add_signal(self, signal_type: str, index: int, value: float, 
                   signal_data: Dict = None):
        """添加信歷"""
        self.signals.append({
            'type': signal_type,
            'index': index,
            'value': value,
            'data': signal_data or {},
            'timestamp': datetime.now(),
        })


class MovingAverageIndicator(BaseIndicator):
    """移動平均指標"""
    
    def __init__(self):
        super().__init__(
            name='Moving Averages',
            description='Simple and Exponential Moving Averages',
            overlay=True
        )
        self.parameters = {
            'sma_length': 20,
            'ema_length': 50,
        }
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df['close']
        
        sma = close.rolling(window=self.parameters['sma_length']).mean()
        ema = close.ewm(span=self.parameters['ema_length'], adjust=False).mean()
        
        self.add_plot('SMA', sma, color='#4169E1', width=1.5)
        self.add_plot('EMA', ema, color='#FF6347', width=1.5)
        
        # 生成信歷
        for i in range(1, len(df)):
            # SMA 金叉
            if not pd.isna(sma.iloc[i]) and not pd.isna(ema.iloc[i]):
                if sma.iloc[i-1] <= ema.iloc[i-1] and sma.iloc[i] > ema.iloc[i]:
                    self.add_signal('bullish_cross', i, close.iloc[i])
                elif sma.iloc[i-1] >= ema.iloc[i-1] and sma.iloc[i] < ema.iloc[i]:
                    self.add_signal('bearish_cross', i, close.iloc[i])
        
        return {
            'plots': self.plots,
            'signals': self.signals,
        }


class RSIIndicator(BaseIndicator):
    """相對強度指數"""
    
    def __init__(self):
        super().__init__(
            name='RSI',
            description='Relative Strength Index',
            overlay=False
        )
        self.parameters = {
            'period': 14,
            'overbought': 70,
            'oversold': 30,
        }
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['period']).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        self.add_plot('RSI', rsi, color='#9370DB', width=1.5)
        self.add_plot('Overbought', pd.Series([self.parameters['overbought']] * len(rsi), index=rsi.index), 
                     color='#FF0000', line_style='--', width=1.0, alpha=0.7)
        self.add_plot('Oversold', pd.Series([self.parameters['oversold']] * len(rsi), index=rsi.index), 
                     color='#00FF00', line_style='--', width=1.0, alpha=0.7)
        
        # 信歷
        for i in range(1, len(rsi)):
            if not pd.isna(rsi.iloc[i]):
                if rsi.iloc[i] > self.parameters['overbought']:
                    self.add_signal('overbought', i, rsi.iloc[i])
                elif rsi.iloc[i] < self.parameters['oversold']:
                    self.add_signal('oversold', i, rsi.iloc[i])
        
        return {
            'plots': self.plots,
            'signals': self.signals,
        }


class MACDIndicator(BaseIndicator):
    """MACD 指標"""
    
    def __init__(self):
        super().__init__(
            name='MACD',
            description='Moving Average Convergence Divergence',
            overlay=False
        )
        self.parameters = {
            'fast': 12,
            'slow': 26,
            'signal': 9,
        }
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df['close']
        
        ema_fast = close.ewm(span=self.parameters['fast'], adjust=False).mean()
        ema_slow = close.ewm(span=self.parameters['slow'], adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.parameters['signal'], adjust=False).mean()
        histogram = macd_line - signal_line
        
        self.add_plot('MACD', macd_line, color='#1E90FF', width=1.5)
        self.add_plot('Signal', signal_line, color='#FF6347', width=1.5)
        
        # 直方圖
        colors = ['#00AA00' if h >= 0 else '#CC0000' for h in histogram]
        for i, color in enumerate(colors):
            if not pd.isna(histogram.iloc[i]):
                self.add_plot(f'Histogram_{i}', pd.Series([histogram.iloc[i]], index=[histogram.index[i]]), 
                             color=color, width=2.0, alpha=0.7)
        
        # 信歷
        for i in range(1, len(macd_line)):
            if not pd.isna(macd_line.iloc[i]) and not pd.isna(signal_line.iloc[i]):
                if macd_line.iloc[i-1] <= signal_line.iloc[i-1] and macd_line.iloc[i] > signal_line.iloc[i]:
                    self.add_signal('bullish_cross', i, macd_line.iloc[i])
                elif macd_line.iloc[i-1] >= signal_line.iloc[i-1] and macd_line.iloc[i] < signal_line.iloc[i]:
                    self.add_signal('bearish_cross', i, macd_line.iloc[i])
        
        return {
            'plots': self.plots,
            'signals': self.signals,
        }


class BollingerBandsIndicator(BaseIndicator):
    """布林帶指標"""
    
    def __init__(self):
        super().__init__(
            name='Bollinger Bands',
            description='Bollinger Bands with Moving Average',
            overlay=True
        )
        self.parameters = {
            'period': 20,
            'deviation': 2.0,
        }
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df['close']
        
        sma = close.rolling(window=self.parameters['period']).mean()
        std = close.rolling(window=self.parameters['period']).std()
        
        upper = sma + (std * self.parameters['deviation'])
        lower = sma - (std * self.parameters['deviation'])
        
        self.add_plot('Middle', sma, color='#FFD700', width=1.5)
        self.add_plot('Upper', upper, color='#DC143C', line_style='--', width=1.0, alpha=0.7)
        self.add_plot('Lower', lower, color='#32CD32', line_style='--', width=1.0, alpha=0.7)
        
        # 信歷
        for i in range(len(df)):
            if not pd.isna(upper.iloc[i]) and close.iloc[i] > upper.iloc[i]:
                self.add_signal('touch_upper', i, close.iloc[i])
            elif not pd.isna(lower.iloc[i]) and close.iloc[i] < lower.iloc[i]:
                self.add_signal('touch_lower', i, close.iloc[i])
        
        return {
            'plots': self.plots,
            'signals': self.signals,
        }


class IndicatorManager:
    """指標管理器"""
    
    def __init__(self):
        self.indicators = {}
        self.overlay_indicators = []
        self.subplot_indicators = []
    
    def register_indicator(self, indicator: BaseIndicator):
        """註冊指標"""
        self.indicators[indicator.name] = indicator
        
        if indicator.overlay:
            self.overlay_indicators.append(indicator)
        else:
            self.subplot_indicators.append(indicator)
    
    def calculate_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """計算所有指標"""
        results = {}
        
        for name, indicator in self.indicators.items():
            try:
                result = indicator.calculate(df)
                results[name] = result
            except Exception as e:
                print(f'Error calculating {name}: {str(e)}')
        
        return results
    
    def get_signals(self, indicator_name: str) -> List[Dict]:
        """取得信歷"""
        if indicator_name in self.indicators:
            return self.indicators[indicator_name].signals
        return []
    
    def plot_indicators(self, df: pd.DataFrame, fig=None, max_subplots: int = 3):
        """繪費所有指標"""
        if fig is None:
            num_subplots = 1 + len(self.subplot_indicators)
            fig, axes = plt.subplots(num_subplots, 1, figsize=(14, 4*num_subplots), sharex=True)
            if num_subplots == 1:
                axes = [axes]
        else:
            axes = fig.get_axes()
        
        # K線
        ax_main = axes[0]
        self._plot_candlesticks(ax_main, df)
        
        # 絲圖指標
        for indicator in self.overlay_indicators:
            for plot_name, plot_data in indicator.plots.items():
                for plot_item in plot_data:
                    ax_main.plot(plot_item['data'], label=plot_name, color=plot_item['color'],
                               linestyle=plot_item['line_style'], linewidth=plot_item['width'],
                               alpha=plot_item['alpha'])
        
        ax_main.set_title('Price Chart with Indicators')
        ax_main.legend(loc='best')
        ax_main.grid(True, alpha=0.3)
        
        # 副綕指標
        for idx, indicator in enumerate(self.subplot_indicators, start=1):
            if idx < len(axes):
                ax = axes[idx]
                
                for plot_name, plot_data in indicator.plots.items():
                    for plot_item in plot_data:
                        ax.plot(plot_item['data'], label=plot_name, color=plot_item['color'],
                               linestyle=plot_item['line_style'], linewidth=plot_item['width'],
                               alpha=plot_item['alpha'])
                
                ax.set_title(indicator.name)
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """繪K線"""
        display_bars = min(500, len(df))
        df_display = df.iloc[-display_bars:].reset_index(drop=True)
        
        for i in range(len(df_display)):
            o, h, l, c = df_display.loc[i, ['open', 'high', 'low', 'close']]
            color = '#00AA00' if c >= o else '#CC0000'
            
            ax.plot([i, i], [l, h], color=color, linewidth=0.8)
            body_size = abs(c - o) if abs(c - o) > 0 else (h - l) * 0.001
            body_bottom = min(o, c)
            ax.bar(i, body_size, width=0.6, bottom=body_bottom, color=color, alpha=0.9, edgecolor=color)
        
        ax.set_xlabel('Bars')
        ax.set_ylabel('Price')


if __name__ == '__main__':
    # 測試
    manager = IndicatorManager()
    manager.register_indicator(MovingAverageIndicator())
    manager.register_indicator(RSIIndicator())
    manager.register_indicator(MACDIndicator())
    manager.register_indicator(BollingerBandsIndicator())
    
    # 加載數揚
    df = pd.read_parquet('data/btc_15m.parquet')
    
    # 計算所有指標
    results = manager.calculate_all(df)
    
    # 繪費
    fig = manager.plot_indicators(df)
    plt.show()
