import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pickle
import json
from pathlib import Path


class SupplyDemandDetector:
    """Supply/Demand 區域檢測器"""
    def __init__(self, aggregation_factor=15, zone_length=100, 
                 show_supply_zones=True, show_demand_zones=True):
        self.aggregation_factor = aggregation_factor
        self.zone_length = zone_length
        self.show_supply_zones = show_supply_zones
        self.show_demand_zones = show_demand_zones

    def detect_zones(self, df: pd.DataFrame):
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)

        group_start_idx = None
        agg_open = agg_high = agg_low = agg_close = None
        prev_is_bullish = None

        prev_supply_low = None
        prev_supply_high = None
        prev_supply_start = None

        prev_demand_low = None
        prev_demand_high = None
        prev_demand_start = None

        supply_zone_used = False
        demand_zone_used = False

        supply_zones = []
        demand_zones = []

        for i in range(n):
            if group_start_idx is None:
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]
                continue

            agg_high = max(agg_high, h[i])
            agg_low = min(agg_low, l[i])
            agg_close = c[i]

            bars_in_group = i - group_start_idx + 1
            is_new_group = bars_in_group >= self.aggregation_factor

            if is_new_group:
                is_bullish = agg_close >= agg_open

                if prev_is_bullish is not None:
                    if is_bullish and not prev_is_bullish:
                        supply_zone_used = False
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
                        demand_zone_used = False
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

                if is_bullish:
                    prev_demand_low = agg_low
                    prev_demand_high = agg_high
                    prev_demand_start = group_start_idx
                else:
                    prev_supply_low = agg_low
                    prev_supply_high = agg_high
                    prev_supply_start = group_start_idx

                prev_is_bullish = is_bullish
                
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]

        return {'supply': supply_zones, 'demand': demand_zones}


class ModelEnsembleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Crypto 預測系統 + Supply/Demand Zones')
        self.root.geometry('1400x850')
        
        self.df = None
        self.models = {}
        
        # 建立 tab 框架
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: 數據載入
        self.load_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.load_frame, text='數據載入')
        self.setup_load_tab()
        
        # Tab 2: 特徵工程
        self.feature_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.feature_frame, text='特徵工程')
        self.setup_feature_tab()
        
        # Tab 3: 模型訓練
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text='模型訓練')
        self.setup_train_tab()
        
        # Tab 4: 模型評估
        self.eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.eval_frame, text='模型評估')
        self.setup_eval_tab()
        
        # Tab 5: 預測
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text='預測')
        self.setup_predict_tab()
        
        # Tab 6: Supply/Demand Zones
        self.supply_demand_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.supply_demand_frame, text='Supply/Demand Zones')
        self.setup_supply_demand_tab()

    def setup_load_tab(self):
        frame = ttk.LabelFrame(self.load_frame, text='數據載入', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Button(frame, text='載入本機 CSV/Parquet', 
                  command=self.load_local_data).pack(pady=10)
        
        ttk.Button(frame, text='載入預設數據 (data/btc_15m.parquet)', 
                  command=self.load_default_data).pack(pady=10)
        
        self.load_status = ttk.Label(frame, text='未載入數據', foreground='red')
        self.load_status.pack(pady=10)
        
        # 數據統計
        self.load_info = ttk.Label(frame, text='', justify=tk.LEFT)
        self.load_info.pack(pady=10)

    def setup_feature_tab(self):
        frame = ttk.LabelFrame(self.feature_frame, text='特徵工程', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text='功能開發中...').pack(pady=10)
        ttk.Label(frame, text='此 Tab 將用於特徵工程配置和數據預處理').pack()

    def setup_train_tab(self):
        frame = ttk.LabelFrame(self.train_frame, text='模型訓練', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text='功能開發中...').pack(pady=10)
        ttk.Label(frame, text='此 Tab 將用於訓練 LightGBM/XGBoost 模型').pack()

    def setup_eval_tab(self):
        frame = ttk.LabelFrame(self.eval_frame, text='模型評估', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text='功能開發中...').pack(pady=10)
        ttk.Label(frame, text='此 Tab 將用於模型性能評估').pack()

    def setup_predict_tab(self):
        frame = ttk.LabelFrame(self.predict_frame, text='預測', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text='功能開發中...').pack(pady=10)
        ttk.Label(frame, text='此 Tab 將用於執行模型預測').pack()

    def setup_supply_demand_tab(self):
        frame = ttk.Frame(self.supply_demand_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 參數框
        param_frame = ttk.LabelFrame(frame, text='檢測參數', padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text='聚合因子:').pack(side=tk.LEFT, padx=5)
        self.agg_spinbox = ttk.Spinbox(param_frame, from_=5, to=50, width=10)
        self.agg_spinbox.set(15)
        self.agg_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(param_frame, text='Zone 長度:').pack(side=tk.LEFT, padx=5)
        self.zone_length_spinbox = ttk.Spinbox(param_frame, from_=50, to=200, width=10)
        self.zone_length_spinbox.set(100)
        self.zone_length_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(param_frame, text='重新檢測', 
                  command=self.detect_and_plot).pack(side=tk.LEFT, padx=5)
        
        # 圖表框
        chart_frame = ttk.Frame(frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.fig = Figure(figsize=(13, 6), dpi=100)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_local_data(self):
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("Parquet files", "*.parquet"), ("All files", "*.*")]
            )
            if filepath:
                if filepath.endswith('.csv'):
                    self.df = pd.read_csv(filepath)
                else:
                    self.df = pd.read_parquet(filepath)
                
                self.update_load_status()
                messagebox.showinfo('成功', f'已載入 {len(self.df)} 行數據')
        except Exception as e:
            messagebox.showerror('錯誤', f'載入失敗: {str(e)}')

    def load_default_data(self):
        try:
            path = 'data/btc_15m.parquet'
            if not Path(path).exists():
                path = 'btc_15m.parquet'
            
            self.df = pd.read_parquet(path)
            self.update_load_status()
            messagebox.showinfo('成功', f'已載入預設數據: {len(self.df)} 行')
        except Exception as e:
            messagebox.showerror('錯誤', f'無法載入預設數據: {str(e)}')

    def update_load_status(self):
        if self.df is not None:
            self.load_status.config(text=f'已載入: {len(self.df)} 行', foreground='green')
            info_text = f"""數據信息：
行數: {len(self.df)}
列: {', '.join(self.df.columns[:5])}...
時間範圍: {self.df.index[0] if len(self.df) > 0 else '未知'} 到 {self.df.index[-1] if len(self.df) > 0 else '未知'}"""
            self.load_info.config(text=info_text)

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
            self.plot_kline_with_zones(zones, agg_factor)
            
            messagebox.showinfo('成功', 
                f'檢測完成\nSupply Zones: {len(zones["supply"])}\nDemand Zones: {len(zones["demand"])}')
        except Exception as e:
            messagebox.showerror('錯誤', f'檢測失敗: {str(e)}')

    def plot_kline_with_zones(self, zones, agg_factor):
        display_bars = 1000
        df = self.df.iloc[-display_bars:].reset_index(drop=True) if len(self.df) > display_bars else self.df
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # 繪製 K 線
        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]
        
        width = 0.6
        if len(up) > 0:
            ax.bar(up.index, up['close'] - up['open'], width, 
                   bottom=up['open'], color='green', alpha=0.7)
            ax.plot(up.index, up['high'], color='green', linewidth=0.5)
            ax.plot(up.index, up['low'], color='green', linewidth=0.5)
        
        if len(down) > 0:
            ax.bar(down.index, down['close'] - down['open'], width, 
                   bottom=down['open'], color='red', alpha=0.7)
            ax.plot(down.index, down['high'], color='red', linewidth=0.5)
            ax.plot(down.index, down['low'], color='red', linewidth=0.5)
        
        # 繪製 zones
        for zone in zones['supply']:
            ax.fill_between(range(zone['start_idx'], min(zone['end_idx'] + 1, len(df))),
                           zone['low'], zone['high'], 
                           alpha=0.15, color='red')
        
        for zone in zones['demand']:
            ax.fill_between(range(zone['start_idx'], min(zone['end_idx'] + 1, len(df))),
                           zone['low'], zone['high'], 
                           alpha=0.15, color='green')
        
        ax.set_xlabel(f'Bar Index (最近 {min(len(df), display_bars)} 根)')
        ax.set_ylabel('Price (USDT)')
        ax.set_title(f'K線圖 + Supply/Demand Zones (聚合: {agg_factor}, 共 {len(zones["supply"]) + len(zones["demand"])} 個 zones)')
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.chart_canvas.draw()


def main():
    root = tk.Tk()
    app = ModelEnsembleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
