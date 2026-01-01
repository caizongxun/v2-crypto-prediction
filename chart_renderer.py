import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

class ChartRenderer:
    """使用 matplotlib 渲染 K 線圖表且載入指標"""
    
    @staticmethod
    def create_candlestick_chart(df: pd.DataFrame, num_candles: int = 300,
                                 order_blocks: List[Dict] = None,
                                 fib_levels: Dict = None,
                                 title: str = 'Candlestick Chart') -> plt.Figure:
        """
        作成 K 線圖表
        
        Args:
            df: 包含 OHLCV 數據的 DataFrame
            num_candles: 要顯示的蠟燭数（最搞 300根）
            order_blocks: 訂單塊數據
            fib_levels: 斐波那契級別
            title: 圖表標題
            
        Returns:
            matplotlib Figure 物件
        """
        # 选旇最后 N 根 K 線
        start_idx = max(0, len(df) - num_candles)
        df_plot = df.iloc[start_idx:].reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2b2b2b')
        
        # 絫製 K 線
        width = 0.6
        
        for idx, row in df_plot.iterrows():
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # 上下影
            ax.plot([idx, idx], [low_price, high_price], color='white', linewidth=1)
            
            # K 線既体
            color = '#26a69a' if close_price >= open_price else '#ef5350'  # 綠赤
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            
            rect = Rectangle((idx - width/2, bottom), width, max(height, 0.0001),
                            facecolor=color, edgecolor=color, linewidth=0.5)
            ax.add_patch(rect)
        
        # 繫制訂單塊 (紅色看跌，綠色看漲)
        if order_blocks:
            for ob in order_blocks:
                if ob['type'] == 'bullish':
                    # 看漲訂單塊 - 綠色
                    ax.axhspan(ob['low'], ob['high'], alpha=0.2, color='green', linewidth=1)
                    # 平均佋
                    ax.axhline(ob['avg'], color='green', linestyle='--', linewidth=1, alpha=0.7)
                else:  # bearish
                    # 看跌訂單塊 - 紅色
                    ax.axhspan(ob['low'], ob['high'], alpha=0.2, color='red', linewidth=1)
                    # 平均伤
                    ax.axhline(ob['avg'], color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # 繫制斐波那契級別
        if fib_levels:
            colors_fib = {
                '0.0': '#ffffff',
                '0.236': '#aaaaaa',
                '0.382': '#888888',
                '0.5': '#666666',
                '0.618': '#444444',
                '0.786': '#222222',
                '1.0': '#ffffff'
            }
            
            for level_str, price in fib_levels.items():
                color = colors_fib.get(level_str, '#cccccc')
                line_style = '-' if level_str in ['0.0', '1.0'] else '--'
                ax.axhline(price, color=color, linestyle=line_style, linewidth=1, alpha=0.5)
                
                # 標籤
                ax.text(len(df_plot) - 1, price, f'{level_str}', fontsize=8, 
                       color=color, va='center', ha='right')
        
        # 設置行間及標符
        ax.set_xlabel('Bar Index', color='white', fontsize=10)
        ax.set_ylabel('Price', color='white', fontsize=10)
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')
        
        # 算毊規森特残設置
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.1, color='white')
        
        # Y 標符伕段
        y_min = df_plot['low'].min()
        y_max = df_plot['high'].max()
        ax.set_ylim(y_min * 0.99, y_max * 1.01)
        ax.set_xlim(-1, len(df_plot))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig: plt.Figure, filepath: str):
        """保存圖表为文件"""
        fig.savefig(filepath, facecolor='#1a1a1a', dpi=100, bbox_inches='tight')
        plt.close(fig)
