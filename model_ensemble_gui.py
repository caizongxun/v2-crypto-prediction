import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class SupplyDemandDetector:
    def __init__(self, aggregation_factor=15, zone_length=100, 
                 show_supply_zones=True, show_demand_zones=True):
        self.aggregation_factor = aggregation_factor
        self.zone_length = zone_length
        self.show_supply_zones = show_supply_zones
        self.show_demand_zones = show_demand_zones

    def detect_zones(self, df: pd.DataFrame):
        """
        偵測 Supply / Demand 區域
        
        核心邏輯：
        1. 聚合 K 線：以 aggregation_factor 根 bar 為單位聚合
        2. 方向轉變判斷：當聚合 K 線方向改變時（bullish->bearish 或 bearish->bullish）
        3. 只在轉變點生成 zone，防止密集生成
        4. demand zone：在 bearish->bullish 轉變時生成
        5. supply zone：在 bullish->bearish 轉變時生成
        """
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)

        # 聚合組狀態
        group_start_idx = None
        agg_open = agg_high = agg_low = agg_close = None
        prev_is_bullish = None  # 追蹤上一個聚合 K 線是否為 bullish

        # 前一個 zone 的資訊
        prev_supply_low = None
        prev_supply_high = None
        prev_supply_start = None

        prev_demand_low = None
        prev_demand_high = None
        prev_demand_start = None

        # Zone 使用標誌
        supply_zone_used = False
        demand_zone_used = False

        supply_zones = []
        demand_zones = []

        for i in range(n):
            # 初始化第一組
            if group_start_idx is None:
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]
                continue

            # 聚合更新
            agg_high = max(agg_high, h[i])
            agg_low = min(agg_low, l[i])
            agg_close = c[i]

            bars_in_group = i - group_start_idx + 1
            is_new_group = bars_in_group >= self.aggregation_factor

            if is_new_group:
                # 一個聚合 K 線完成
                is_bullish = agg_close >= agg_open

                # 方向改變判斷
                if prev_is_bullish is not None:
                    if is_bullish and not prev_is_bullish:
                        # bearish -> bullish 轉變
                        supply_zone_used = False  # 重設供應區使用標誌
                        
                        # 生成前一個 bearish 週期的 demand zone
                        if prev_demand_high is not None and not demand_zone_used and self.show_demand_zones:
                            zone_start = prev_demand_start
                            zone_end = min(prev_demand_start + self.zone_length, n - 1)
                            demand_zones.append({
                                'start_idx': zone_start,
                                'end_idx': zone_end,
                                'high': prev_demand_high,
                                'low': prev_demand_low,
                            })
                            demand_zone_used = True

                    elif not is_bullish and prev_is_bullish:
                        # bullish -> bearish 轉變
                        demand_zone_used = False  # 重設需求區使用標誌
                        
                        # 生成前一個 bullish 週期的 supply zone
                        if prev_supply_low is not None and not supply_zone_used and self.show_supply_zones:
                            zone_start = prev_supply_start
                            zone_end = min(prev_supply_start + self.zone_length, n - 1)
                            supply_zones.append({
                                'start_idx': zone_start,
                                'end_idx': zone_end,
                                'high': prev_supply_high,
                                'low': prev_supply_low,
                            })
                            supply_zone_used = True

                # 記錄當前週期為下次使用
                if is_bullish:
                    prev_demand_low = agg_low
                    prev_demand_high = agg_high
                    prev_demand_start = group_start_idx
                else:
                    prev_supply_low = agg_low
                    prev_supply_high = agg_high
                    prev_supply_start = group_start_idx

                # 追蹤目前方向
                prev_is_bullish = is_bullish
                
                # 新聚合組開始
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]

        return {
            'supply': supply_zones,
            'demand': demand_zones,
        }


class KlineCanvas:
    def __init__(self, fig, display_bars=1000):
        self.fig = fig
        self.display_bars = display_bars
        self.detector = SupplyDemandDetector(aggregation_factor=15, zone_length=100)

    def plot_kline_with_zones(self, df: pd.DataFrame, zones: dict):
        # 只顯示最後 display_bars 根
        if len(df) > self.display_bars:
            df = df.iloc[-self.display_bars:].reset_index(drop=True)
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # 繪製 K 線
        width = 0.6
        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]
        
        # 上升 K 線（綠色）
        if len(up) > 0:
            ax.bar(up.index, up['close'] - up['open'], width, 
                   bottom=up['open'], color='green', alpha=0.7)
            ax.plot(up.index, up['high'], color='green', linewidth=0.5)
            ax.plot(up.index, up['low'], color='green', linewidth=0.5)
        
        # 下降 K 線（紅色）
        if len(down) > 0:
            ax.bar(down.index, down['close'] - down['open'], width, 
                   bottom=down['open'], color='red', alpha=0.7)
            ax.plot(down.index, down['high'], color='red', linewidth=0.5)
            ax.plot(down.index, down['low'], color='red', linewidth=0.5)
        
        # 繪製 Supply/Demand Zones
        # Supply zones (紅色背景)
        for zone in zones['supply']:
            ax.axvspan(zone['start_idx'], zone['end_idx'], 
                       ymin=0, ymax=1, alpha=0.2, color='red')
            ax.fill_between(range(zone['start_idx'], min(zone['end_idx'] + 1, len(df))),
                           zone['low'], zone['high'], 
                           alpha=0.15, color='red', label='Supply' if zone == zones['supply'][0] else '')
        
        # Demand zones (綠色背景)
        for zone in zones['demand']:
            ax.axvspan(zone['start_idx'], zone['end_idx'], 
                       ymin=0, ymax=1, alpha=0.2, color='green')
            ax.fill_between(range(zone['start_idx'], min(zone['end_idx'] + 1, len(df))),
                           zone['low'], zone['high'], 
                           alpha=0.15, color='green', label='Demand' if zone == zones['demand'][0] else '')
        
        ax.set_xlabel('Bar Index (最近 {} 根)'.format(min(len(df), self.display_bars)))
        ax.set_ylabel('Price (USDT)')
        ax.set_title('K線圖 Supply/Demand Zones - 共 {} 個 zones (Supply: {}, Demand: {})'.format(
            len(zones['supply']) + len(zones['demand']),
            len(zones['supply']),
            len(zones['demand'])
        ))
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()


class ModelEnsembleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Crypto Prediction Model Ensemble + Supply/Demand Zones')
        self.root.geometry('1400x800')
        
        self.df = None
        self.zones = None
        
        # 建立 tab
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: 數據載入
        self.load_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.load_frame, text='數據載入')
        self.setup_load_tab()
        
        # Tab 2: Supply/Demand Visualization
        self.supply_demand_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.supply_demand_frame, text='Supply/Demand Zones')
        self.setup_supply_demand_tab()

    def setup_load_tab(self):
        frame = ttk.LabelFrame(self.load_frame, text='數據載入選項', padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text='選擇數據源：').pack()
        
        ttk.Button(frame, text='載入本地 CSV 或 Parquet', 
                  command=self.load_local_data).pack(pady=10)
        
        self.status_label = ttk.Label(frame, text='未載入數據', foreground='red')
        self.status_label.pack()

    def setup_supply_demand_tab(self):
        frame = ttk.Frame(self.supply_demand_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 參數調整
        param_frame = ttk.LabelFrame(frame, text='檢測參數', padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text='聚合因子 (Aggregation):').pack(side=tk.LEFT, padx=5)
        self.agg_spinbox = ttk.Spinbox(param_frame, from_=5, to=50, width=10)
        self.agg_spinbox.set(15)
        self.agg_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(param_frame, text='Zone 長度:').pack(side=tk.LEFT, padx=5)
        self.zone_length_spinbox = ttk.Spinbox(param_frame, from_=50, to=200, width=10)
        self.zone_length_spinbox.set(100)
        self.zone_length_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(param_frame, text='重新檢測', 
                  command=self.detect_and_plot).pack(side=tk.LEFT, padx=5)
        
        # 圖表
        self.canvas_frame = ttk.Frame(frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.fig = Figure(figsize=(13, 6), dpi=100)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_local_data(self):
        try:
            # 嘗試載入 CSV
            df = pd.read_csv('data/btc_15m.csv')
            self.df = df
            self.status_label.config(text=f'已載入數據: {len(df)} 行', foreground='green')
            messagebox.showinfo('成功', f'載入 {len(df)} 行數據')
        except:
            try:
                # 嘗試載入 Parquet
                df = pd.read_parquet('data/btc_15m.parquet')
                self.df = df
                self.status_label.config(text=f'已載入數據: {len(df)} 行', foreground='green')
                messagebox.showinfo('成功', f'載入 {len(df)} 行數據')
            except Exception as e:
                messagebox.showerror('錯誤', f'無法載入數據: {str(e)}')

    def detect_and_plot(self):
        if self.df is None:
            messagebox.showwarning('警告', '請先載入數據')
            return
        
        try:
            agg_factor = int(self.agg_spinbox.get())
            zone_length = int(self.zone_length_spinbox.get())
            
            detector = SupplyDemandDetector(
                aggregation_factor=agg_factor,
                zone_length=zone_length
            )
            
            zones = detector.detect_zones(self.df)
            self.zones = zones
            
            canvas = KlineCanvas(self.fig, display_bars=1000)
            canvas.plot_kline_with_zones(self.df, zones)
            self.chart_canvas.draw()
            
            messagebox.showinfo('成功', 
                f'檢測完成\nSupply Zones: {len(zones["supply"])}\nDemand Zones: {len(zones["demand"])}')
        except Exception as e:
            messagebox.showerror('錯誤', f'檢測失敗: {str(e)}')


def main():
    root = tk.Tk()
    app = ModelEnsembleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
