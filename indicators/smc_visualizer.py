import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from typing import List, Optional, Tuple
from .smc import SmartMoneyStructure, Zone, Structure


class SMCVisualizer:
    """SMC 指標可視化工具"""

    def __init__(self, figsize: Tuple[int, int] = (16, 8), dpi: int = 100):
        """
        初始化可視化工具
        
        Args:
            figsize: 圖表大小
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax_price = None
        self.ax_volume = None

    def plot(self, df: pd.DataFrame, smc: SmartMoneyStructure, 
             start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> plt.Figure:
        """
        繪製 K 線圖表和 SMC zones
        
        Args:
            df: K 線數據
            smc: SMC 分析對象
            start_idx: 開始索引 (預設: -500)
            end_idx: 結束索引 (預設: -1)
            
        Returns:
            matplotlib Figure
        """
        # 設定時間窗口
        if start_idx is None:
            start_idx = max(0, len(df) - 500)
        if end_idx is None:
            end_idx = len(df)
            
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # 調整 zone 的索引
        adjusted_zones = []
        for zone in smc.zones:
            if start_idx <= zone.created_at_idx < end_idx:
                adjusted_zone = Zone(
                    bar_index=zone.bar_index - start_idx,
                    high=zone.high,
                    low=zone.low,
                    structure_type=zone.structure_type,
                    created_at_idx=zone.created_at_idx - start_idx
                )
                adjusted_zones.append(adjusted_zone)
        
        # 創建圖表
        self.fig, (self.ax_price, self.ax_volume) = plt.subplots(
            2, 1, figsize=self.figsize, dpi=self.dpi,
            gridspec_kw={'height_ratios': [3, 1]}
        )
        
        # 繪製 K 線
        self._plot_klines(window_df, self.ax_price)
        
        # 繪製 zones
        self._plot_zones(adjusted_zones, window_df, self.ax_price)
        
        # 繪製成交量
        self._plot_volume(window_df, self.ax_volume)
        
        # 格式化
        self.ax_price.set_title('BTC/USDT - Smart Money Concept', fontsize=14, fontweight='bold')
        self.ax_price.grid(True, alpha=0.3)
        self.ax_volume.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self.fig

    def _plot_klines(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        繪製 K 線
        
        Args:
            df: K 線數據
            ax: matplotlib 坐標軸
        """
        # 準備顏色
        colors = np.where(df['close'] >= df['open'], 'green', 'red')
        
        # 繪製 wick (上下影線)
        for i in range(len(df)):
            ax.plot([i, i], [df.iloc[i]['low'], df.iloc[i]['high']], 
                   color=colors[i], linewidth=0.5, alpha=0.8)
        
        # 繪製 body (實體)
        width = 0.6
        for i in range(len(df)):
            open_price = df.iloc[i]['open']
            close_price = df.iloc[i]['close']
            high = max(open_price, close_price)
            low = min(open_price, close_price)
            
            ax.add_patch(Rectangle(
                (i - width/2, low), width, high - low,
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=0.8
            ))
        
        # 設定 X 軸
        ax.set_xlim(-1, len(df))
        ax.set_ylim(df['low'].min() * 0.98, df['high'].max() * 1.02)
        ax.set_ylabel('Price (USDT)', fontsize=11)

    def _plot_zones(self, zones: List[Zone], df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        繪製 Supply/Demand zones
        
        Args:
            zones: Zone 列表
            df: K 線數據 (用於設定 Y 軸範圍)
            ax: matplotlib 坐標軸
        """
        for zone in zones:
            # 決定顏色: Supply (紅色), Demand (綠色)
            color = 'rgba(255, 0, 0, 0.1)' if zone.is_supply else 'rgba(0, 255, 0, 0.1)'
            edge_color = 'red' if zone.is_supply else 'green'
            label = 'Supply' if zone.is_supply else 'Demand'
            
            # 繪製區域 (從生成位置到圖表末端)
            zone_height = zone.high - zone.low
            rect = Rectangle(
                (zone.created_at_idx, zone.low),
                len(df) - zone.created_at_idx,
                zone_height,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=1.5,
                linestyle='--',
                alpha=0.3,
                label=label if zone == zones[0] else None
            )
            ax.add_patch(rect)
            
            # 添加標籤
            mid_price = zone.mid
            ax.text(
                zone.created_at_idx + 2,
                mid_price,
                f"{label}\n{zone.low:.0f}-{zone.high:.0f}",
                fontsize=8,
                color=edge_color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=edge_color)
            )

    def _plot_volume(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        繪製成交量
        
        Args:
            df: K 線數據
            ax: matplotlib 坐標軸
        """
        colors = np.where(df['close'] >= df['open'], 'green', 'red')
        ax.bar(range(len(df)), df['volume'], color=colors, alpha=0.5, width=0.8)
        ax.set_ylabel('Volume', fontsize=11)
        ax.set_xlabel('Bar Index', fontsize=11)
        ax.set_xlim(-1, len(df))

    def plot_zones_only(self, zones: List[Zone], price_range: Tuple[float, float],
                        num_bars: int = 500) -> plt.Figure:
        """
        繪製僅包含 zones 信息的圖表
        
        Args:
            zones: Zone 列表
            price_range: (最低價, 最高價)
            num_bars: 顯示的 bar 數量
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        min_price, max_price = price_range
        
        for zone in zones:
            color = 'red' if zone.is_supply else 'green'
            label = 'Supply' if zone.is_supply else 'Demand'
            
            ax.axhspan(zone.low, zone.high, color=color, alpha=0.2, label=label)
            ax.plot([zone.created_at_idx, num_bars], [zone.mid, zone.mid], 
                   color=color, linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_xlim(0, num_bars)
        ax.set_ylim(min_price * 0.98, max_price * 1.02)
        ax.set_xlabel('Bar Index', fontsize=11)
        ax.set_ylabel('Price (USDT)', fontsize=11)
        ax.set_title('SMC Zones Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def save(self, filepath: str) -> None:
        """
        保存圖表
        
        Args:
            filepath: 保存路徑
        """
        if self.fig is None:
            raise ValueError("No figure to save. Call plot() first.")
        
        self.fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"Figure saved to {filepath}")

    def show(self) -> None:
        """
        顯示圖表
        """
        if self.fig is None:
            raise ValueError("No figure to show. Call plot() first.")
        
        plt.show()


def create_smc_report(df: pd.DataFrame, smc: SmartMoneyStructure, 
                      output_dir: str = './smc_reports') -> dict:
    """
    生成完整的 SMC 分析報告
    
    Args:
        df: K 線數據
        smc: SMC 分析對象
        output_dir: 輸出目錄
        
    Returns:
        報告字典
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'total_zones': len(smc.zones),
        'supply_zones': sum(1 for z in smc.zones if z.is_supply),
        'demand_zones': sum(1 for z in smc.zones if z.is_demand),
        'zones_data': smc.get_zones_df().to_dict('records') if smc.zones else [],
        'total_legs': len(smc.legs),
        'total_pivots': len(smc.pivots)
    }
    
    # 繪製圖表
    visualizer = SMCVisualizer(figsize=(18, 10))
    visualizer.plot(df, smc)
    visualizer.save(os.path.join(output_dir, 'smc_analysis.png'))
    
    # 保存報告
    import json
    report_path = os.path.join(output_dir, 'smc_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nSMC Report:")
    print(f"  Total Zones: {report['total_zones']}")
    print(f"  Supply Zones: {report['supply_zones']}")
    print(f"  Demand Zones: {report['demand_zones']}")
    print(f"  Report saved to {report_path}")
    
    return report
