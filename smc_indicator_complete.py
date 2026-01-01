"""
Smart Money Concepts Indicator - LuxAlgo Version
轉換自 PineScript，包含完整的圖表箸図遯輯

功能:
- 訂單區塊 (Order Blocks) 檢測
- 結構分析 (BOS/CHoCH)
- 副东黿一平或低 (EQH/EQL)
- Fair Value Gaps 檢測
- 即時標記沒有佔影椲祉名稱
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Pivot:
    """Pivot 資訊非"""
    bar_index: int
    bar_time: int
    current_level: float
    pivot_type: str  # 'HH', 'HL', 'LL', 'LH'


@dataclass
class OrderBlock:
    """Order Block 資訊非"""
    start_idx: int
    end_idx: int
    high: float
    low: float
    block_type: str  # 'bullish' or 'bearish'
    is_mitigated: bool = False
    mitigation_idx: Optional[int] = None


@dataclass
class Structure:
    """結構資訊非"""
    bar_index: int
    structure_type: str  # 'BOS' or 'CHoCH'
    direction: str  # 'bullish' or 'bearish'
    level: float


class SmartMoneyConceptsLuxAlgo:
    """
LuxAlgo 版 Smart Money Concepts 指標
    """
    
    # 常數定義
    BULLISH_LEG = 1
    BEARISH_LEG = 0
    
    BULLISH = 1
    BEARISH = -1
    
    # 颜色
    GREEN = '#089981'
    RED = '#F23645'
    BLUE = '#2157f3'
    GRAY = '#878b94'
    MONO_BULLISH = '#b2b5be'
    MONO_BEARISH = '#5d606b'
    
    def __init__(self, df: pd.DataFrame, swing_length: int = 50, 
                 internal_length: int = 5, filter_confluence: bool = False):
        """
        初始化 SMC 指標
        
        Args:
            df: OHLC 數據框
            swing_length: 擺幅長度 (檢測主要擺幅)
            internal_length: 內部結構擺幅
            filter_confluence: 是否週甑沛段有效性
        """
        self.df = df.copy()
        self.swing_length = swing_length
        self.internal_length = internal_length
        self.filter_confluence = filter_confluence
        
        # 組光數組
        self.parsed_highs = []
        self.parsed_lows = []
        self.times = []
        
        # 結果存儲
        self.pivots_high: List[Pivot] = []
        self.pivots_low: List[Pivot] = []
        self.order_blocks: List[OrderBlock] = []
        self.structures: List[Structure] = []
        self.equal_highs: List[Tuple[int, float]] = []
        self.equal_lows: List[Tuple[int, float]] = []
        
    def leg(self, size: int) -> np.ndarray:
        """
        計算當前的加載 (BULLISH_LEG=1 或 BEARISH_LEG=0)
        """
        n = len(self.df)
        leg_array = np.zeros(n)
        
        for i in range(size, n):
            if i == size:
                # 初始化
                window_high = self.df['high'].iloc[:i].max()
                window_low = self.df['low'].iloc[:i].min()
                
                if self.df['high'].iloc[i] > window_high:
                    leg_array[i] = self.BEARISH_LEG
                elif self.df['low'].iloc[i] < window_low:
                    leg_array[i] = self.BULLISH_LEG
                else:
                    leg_array[i] = leg_array[i-1] if i > 0 else self.BULLISH_LEG
            else:
                # 續繉是否轉換加載
                window_high = self.df['high'].iloc[max(0, i-size):i].max()
                window_low = self.df['low'].iloc[max(0, i-size):i].min()
                
                if self.df['high'].iloc[i] > window_high:
                    leg_array[i] = self.BEARISH_LEG
                elif self.df['low'].iloc[i] < window_low:
                    leg_array[i] = self.BULLISH_LEG
                else:
                    leg_array[i] = leg_array[i-1]
        
        return leg_array
    
    def detect_pivots(self, leg_array: np.ndarray) -> Tuple[List[Pivot], List[Pivot]]:
        """
        檢測樞紐點 (HH, HL, LL, LH)
        """
        n = len(self.df)
        highs = []
        lows = []
        
        last_high_price = None
        last_low_price = None
        
        for i in range(self.swing_length + 1, n):
            # 棂查是否柢化轉換
            if i > 0 and leg_array[i] != leg_array[i-1]:
                
                # BEARISH 轉 BULLISH (低樞紐點)
                if leg_array[i-1] == self.BEARISH_LEG and leg_array[i] == self.BULLISH_LEG:
                    # 找前一段的最低點
                    search_start = max(self.swing_length, i - self.swing_length - 10)
                    min_idx = self.df['low'].iloc[search_start:i].idxmin()
                    min_price = self.df['low'].loc[min_idx]
                    
                    # 判斷是 LL 還是 HL
                    if last_low_price is None:
                        pivot_type = 'LL'
                    elif min_price < last_low_price:
                        pivot_type = 'LL'
                    else:
                        pivot_type = 'HL'
                    
                    lows.append(Pivot(
                        bar_index=min_idx,
                        bar_time=self.df.index[min_idx],
                        current_level=min_price,
                        pivot_type=pivot_type
                    ))
                    last_low_price = min_price
                
                # BULLISH 轉 BEARISH (高樞紐點)
                elif leg_array[i-1] == self.BULLISH_LEG and leg_array[i] == self.BEARISH_LEG:
                    # 找前一段的最高點
                    search_start = max(self.swing_length, i - self.swing_length - 10)
                    max_idx = self.df['high'].iloc[search_start:i].idxmax()
                    max_price = self.df['high'].loc[max_idx]
                    
                    # 判斷是 HH 還是 LH
                    if last_high_price is None:
                        pivot_type = 'HH'
                    elif max_price > last_high_price:
                        pivot_type = 'HH'
                    else:
                        pivot_type = 'LH'
                    
                    highs.append(Pivot(
                        bar_index=max_idx,
                        bar_time=self.df.index[max_idx],
                        current_level=max_price,
                        pivot_type=pivot_type
                    ))
                    last_high_price = max_price
        
        return highs, lows
    
    def detect_order_blocks(self, pivots_high: List[Pivot], 
                           pivots_low: List[Pivot]) -> List[OrderBlock]:
        """
        檢測訂單區塊 (Order Blocks)
        
        规则:
        - 看跌 OB: 从 HH 到下一个 LL（擏去一段了)
        - 看漲 OB: 从 LL 到下一个 HH
        """
        blocks = []
        n = len(self.df)
        
        # 看跌 Order Block (HH -> LL)
        for i, pivot_hh in enumerate(pivots_high):
            if pivot_hh.pivot_type == 'HH':
                # 找下一个 LL
                next_ll = None
                for pivot_ll in pivots_low:
                    if pivot_ll.bar_index > pivot_hh.bar_index:
                        next_ll = pivot_ll
                        break
                
                if next_ll:
                    start = pivot_hh.bar_index
                    end = next_ll.bar_index
                    
                    segment_high = self.df['high'].iloc[start:end+1].max()
                    segment_low = self.df['low'].iloc[start:end+1].min()
                    
                    width = end - start
                    height_pct = ((segment_high - segment_low) / segment_high * 100) if segment_high > 0 else 0
                    
                    # 檢查是否是有效 OB (宽度 5-500, 高度 0.05-20%)
                    if 5 <= width <= 500 and 0.05 <= height_pct <= 20:
                        blocks.append(OrderBlock(
                            start_idx=start,
                            end_idx=end,
                            high=segment_high,
                            low=segment_low,
                            block_type='bearish'
                        ))
        
        # 看漲 Order Block (LL -> HH)
        for i, pivot_ll in enumerate(pivots_low):
            if pivot_ll.pivot_type == 'LL':
                # 找下一个 HH
                next_hh = None
                for pivot_hh in pivots_high:
                    if pivot_hh.bar_index > pivot_ll.bar_index:
                        next_hh = pivot_hh
                        break
                
                if next_hh:
                    start = pivot_ll.bar_index
                    end = next_hh.bar_index
                    
                    segment_high = self.df['high'].iloc[start:end+1].max()
                    segment_low = self.df['low'].iloc[start:end+1].min()
                    
                    width = end - start
                    height_pct = ((segment_high - segment_low) / segment_high * 100) if segment_high > 0 else 0
                    
                    if 5 <= width <= 500 and 0.05 <= height_pct <= 20:
                        blocks.append(OrderBlock(
                            start_idx=start,
                            end_idx=end,
                            high=segment_high,
                            low=segment_low,
                            block_type='bullish'
                        ))
        
        return blocks
    
    def detect_structures(self, pivots_high: List[Pivot], 
                         pivots_low: List[Pivot]) -> List[Structure]:
        """
        檢測結構 (BOS = Break of Structure, CHoCH = Change of Character)
        """
        structures = []
        n = len(self.df)
        
        # 整合所有樞紐點
        all_pivots = []
        for p in pivots_high:
            all_pivots.append(('high', p))
        for p in pivots_low:
            all_pivots.append(('low', p))
        all_pivots.sort(key=lambda x: x[1].bar_index)
        
        # 棂查是否打破結構
        for i in range(1, len(all_pivots)):
            curr_type, curr_pivot = all_pivots[i]
            prev_type, prev_pivot = all_pivots[i-1]
            
            # Bullish BOS/CHoCH
            if curr_type == 'high' and prev_type == 'low':
                pivot_level = curr_pivot.current_level
                
                # 棂查是否打破
                for check_idx in range(curr_pivot.bar_index + 1, 
                                      min(curr_pivot.bar_index + 20, n)):
                    if (self.df['close'].iloc[check_idx] > pivot_level and 
                        self.df['close'].iloc[check_idx-1] <= pivot_level):
                        
                        struct_type = 'CHoCH' if curr_pivot.pivot_type == 'HH' else 'BOS'
                        structures.append(Structure(
                            bar_index=check_idx,
                            structure_type=struct_type,
                            direction='bullish',
                            level=pivot_level
                        ))
                        break
            
            # Bearish BOS/CHoCH
            elif curr_type == 'low' and prev_type == 'high':
                pivot_level = curr_pivot.current_level
                
                for check_idx in range(curr_pivot.bar_index + 1,
                                      min(curr_pivot.bar_index + 20, n)):
                    if (self.df['close'].iloc[check_idx] < pivot_level and
                        self.df['close'].iloc[check_idx-1] >= pivot_level):
                        
                        struct_type = 'CHoCH' if curr_pivot.pivot_type == 'LL' else 'BOS'
                        structures.append(Structure(
                            bar_index=check_idx,
                            structure_type=struct_type,
                            direction='bearish',
                            level=pivot_level
                        ))
                        break
        
        return structures
    
    def analyze(self) -> Dict:
        """
        執行完整的 SMC 分析
        """
        # 標步 1: 計算加載
        leg_swing = self.leg(self.swing_length)
        leg_internal = self.leg(self.internal_length)
        
        # 標步 2: 檢測樞紐點
        pivots_high_swing, pivots_low_swing = self.detect_pivots(leg_swing)
        pivots_high_internal, pivots_low_internal = self.detect_pivots(leg_internal)
        
        # 標步 3: 檢測訂單區塊
        order_blocks = self.detect_order_blocks(pivots_high_swing, pivots_low_swing)
        
        # 標步 4: 檢測結構
        structures = self.detect_structures(pivots_high_swing, pivots_low_swing)
        
        # 標步 5: 檢測等戰件
        self.detect_equal_highs_lows(pivots_high_swing, pivots_low_swing)
        
        return {
            'pivots_high_swing': pivots_high_swing,
            'pivots_low_swing': pivots_low_swing,
            'pivots_high_internal': pivots_high_internal,
            'pivots_low_internal': pivots_low_internal,
            'order_blocks': order_blocks,
            'structures': structures,
            'equal_highs': self.equal_highs,
            'equal_lows': self.equal_lows,
            'leg_swing': leg_swing,
            'leg_internal': leg_internal,
        }
    
    def detect_equal_highs_lows(self, pivots_high: List[Pivot], 
                               pivots_low: List[Pivot],
                               length: int = 20, threshold: float = 0.001):
        """
        檢測副东黿一平或低 (EQH/EQL)
        """
        # 檢查高樞紐
        for i, pivot1 in enumerate(pivots_high):
            for j in range(i + 1, len(pivots_high)):
                pivot2 = pivots_high[j]
                diff = abs(pivot1.current_level - pivot2.current_level)
                if diff / pivot1.current_level < threshold:
                    self.equal_highs.append((pivot2.bar_index, pivot2.current_level))
        
        # 檢查低樞紐
        for i, pivot1 in enumerate(pivots_low):
            for j in range(i + 1, len(pivots_low)):
                pivot2 = pivots_low[j]
                diff = abs(pivot1.current_level - pivot2.current_level)
                if diff / pivot1.current_level < threshold:
                    self.equal_lows.append((pivot2.bar_index, pivot2.current_level))


def plot_smc_analysis(df: pd.DataFrame, analysis_result: Dict, 
                     figsize: Tuple[int, int] = (16, 8)):
    """
    繮譖 SMC 指標分析
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # 患鬼
    n = len(df)
    display_bars = min(500, n)
    offset = n - display_bars
    display_df = df.iloc[-display_bars:].reset_index(drop=True)
    
    price_min = display_df['low'].min()
    price_max = display_df['high'].max()
    price_range = price_max - price_min
    
    # 1. 繫绘 Order Blocks
    for ob in analysis_result['order_blocks']:
        display_start = ob.start_idx - offset
        display_end = ob.end_idx - offset
        
        if display_end >= 0 and display_start < len(display_df):
            plot_start = max(0, display_start)
            plot_end = min(len(display_df) - 1, display_end)
            width = plot_end - plot_start + 1
            
            if ob.is_mitigated:
                color = '#FF6B9D' if ob.block_type == 'bearish' else '#FFB3D9'
                alpha = 0.2
            else:
                color = '#4169E1' if ob.block_type == 'bearish' else '#32CD32'
                alpha = 0.15
            
            rect = Rectangle(
                (plot_start - 0.5, ob.low),
                width,
                ob.high - ob.low,
                linewidth=1.5,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                zorder=2
            )
            ax.add_patch(rect)
    
    # 2. 繫绘髪線 (Candlesticks)
    for i in range(len(display_df)):
        o, h, l, c = display_df.loc[i, ['open', 'high', 'low', 'close']]
        color = '#00AA00' if c >= o else '#CC0000'
        
        # 高低線
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=3, alpha=0.8)
        
        # 庫高線
        body_size = abs(c - o) if abs(c - o) > 0 else price_range * 0.001
        body_bottom = min(o, c)
        ax.bar(i, body_size, width=0.6, bottom=body_bottom,
               color=color, alpha=0.9, edgecolor=color, linewidth=0.5, zorder=3)
    
    # 3. 繫绘樞紐點
    # 高樞紐
    for p in analysis_result['pivots_high_swing']:
        idx = p.bar_index - offset
        if 0 <= idx < len(display_df):
            ax.plot(idx, p.current_level, marker='^', color='#8B0000', 
                   markersize=8, zorder=5, markeredgewidth=1, markeredgecolor='darkred')
            
            # 標記
            pivot_label = f"{p.pivot_type}"
            ax.text(idx, p.current_level + price_range * 0.03, pivot_label,
                   fontsize=7, ha='center', color='#8B0000', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # 低樞紐
    for p in analysis_result['pivots_low_swing']:
        idx = p.bar_index - offset
        if 0 <= idx < len(display_df):
            ax.plot(idx, p.current_level, marker='v', color='#006400',
                   markersize=8, zorder=5, markeredgewidth=1, markeredgecolor='darkgreen')
            
            pivot_label = f"{p.pivot_type}"
            ax.text(idx, p.current_level - price_range * 0.03, pivot_label,
                   fontsize=7, ha='center', color='#006400', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    # 4. 繫绘結構戓破
    for struct in analysis_result['structures']:
        idx = struct.bar_index - offset
        if 0 <= idx < len(display_df):
            if struct.structure_type == 'CHoCH':
                color = '#00CED1'
                linestyle = '--'
                linewidth = 2
            else:  # BOS
                color = '#FFD700'
                linestyle = '-'
                linewidth = 1.5
            
            ax.axvline(x=idx, color=color, linewidth=linewidth, 
                      alpha=0.6, linestyle=linestyle, zorder=4)
            
            # 標記
            struct_label = struct.structure_type
            ax.text(idx, struct.level, f" {struct_label}",
                   fontsize=6, color=color, fontweight='bold', alpha=0.8)
    
    # 5. 繫绘等戰件
    # EQH
    for idx, level in analysis_result['equal_highs']:
        plot_idx = idx - offset
        if 0 <= plot_idx < len(display_df):
            ax.axhline(y=level, color='#FF69B4', linewidth=1, 
                      alpha=0.4, linestyle=':', zorder=1)
    
    # EQL
    for idx, level in analysis_result['equal_lows']:
        plot_idx = idx - offset
        if 0 <= plot_idx < len(display_df):
            ax.axhline(y=level, color='#00FF7F', linewidth=1,
                      alpha=0.4, linestyle=':', zorder=1)
    
    # 图例
    legend_elements = [
        mpatches.Patch(color='#4169E1', alpha=0.15, label='Bearish OB (HH->LL)'),
        mpatches.Patch(color='#32CD32', alpha=0.15, label='Bullish OB (LL->HH)'),
        mpatches.Patch(color='#00AA00', label='Bullish Candle'),
        mpatches.Patch(color='#CC0000', label='Bearish Candle'),
        mpatches.Patch(color='#FFD700', alpha=0.6, label='BOS (Break of Structure)'),
        mpatches.Patch(color='#00CED1', alpha=0.6, label='CHoCH (Change of Character)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # 配置坐標轴
    ax.set_xlabel(f'Bar Index (Last {display_bars} bars)', fontsize=10)
    ax.set_ylabel('Price (USDT)', fontsize=10)
    ax.set_title(f'Smart Money Concepts (LuxAlgo) - Swing Length: {50}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-1, len(display_df))
    ax.set_ylim(price_min - price_range * 0.1, price_max + price_range * 0.1)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    # 示例使用
    import yfinance as yf
    
    # 加載數據
    df = yf.download('BTC-USD', period='1y', interval='1d')
    df.columns = [col.lower() for col in df.columns]
    
    # 扩展列名
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # 执行 SMC 分析
    smc = SmartMoneyConceptsLuxAlgo(df, swing_length=50, internal_length=5)
    result = smc.analyze()
    
    # 繫绘
    fig, ax = plot_smc_analysis(df, result)
    plt.show()
    
    # 打印統計
    print(f"\n=== Smart Money Concepts 分析 ===")
    print(f"高樞紐: {len(result['pivots_high_swing'])}")
    print(f"低樞紐: {len(result['pivots_low_swing'])}")
    print(f"訂單區塊: {len(result['order_blocks'])}")
    print(f"結構打破: {len(result['structures'])}")
