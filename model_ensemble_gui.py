import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib
import pickle
import json
from pathlib import Path

# Set matplotlib font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100


class SupplyDemandDetector:
    """Supply/Demand Zone Detector - follows Pine Script logic exactly"""
    def __init__(self, aggregation_factor=4, zone_length=50):
        self.aggregation_factor = aggregation_factor
        self.zone_length = zone_length

    def detect_zones(self, df: pd.DataFrame):
        """
        Detect supply/demand zones following exact Pine Script logic:
        1. Aggregate N candles into 1 synthetic candle
        2. ONLY when synthetic candle direction CHANGES from previous:
           - If bullish now (was bearish before): create DEMAND zone from previous period
           - If bearish now (was bullish before): create SUPPLY zone from previous period
        3. Zone creation is gated by checking if close crossed the previous range
        4. Use 'used' flag to prevent duplicate zone creation for same reversal
        5. Track if zone is mitigated (touched by price)
        """
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)

        # Aggregation state
        group_start_idx = None
        agg_open = agg_high = agg_low = agg_close = None
        prev_is_bullish = None  # Previous synthetic candle direction

        # State for PREVIOUS period (used when direction changes)
        prev_low = None
        prev_high = None
        prev_start_bar = None
        prev_direction = None  # 'bullish' or 'bearish'

        # Flags to prevent multiple zone creations for same reversal
        supply_zone_used = False  # Prevents multiple supply zones from same bearish->bullish
        demand_zone_used = False  # Prevents multiple demand zones from same bullish->bearish

        supply_zones = []
        demand_zones = []

        for i in range(n):
            # Initialize first group
            if group_start_idx is None:
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]
                continue

            # Update aggregation
            agg_high = max(agg_high, h[i])
            agg_low = min(agg_low, l[i])
            agg_close = c[i]

            bars_in_group = i - group_start_idx + 1
            is_new_group = bars_in_group >= self.aggregation_factor

            if is_new_group:
                # One synthetic candle is complete, determine its direction
                is_bullish = agg_close >= agg_open

                # === DIRECTION CHANGE DETECTION ===
                # Only create zones when synthetic candle direction CHANGES
                if prev_is_bullish is not None:
                    if is_bullish and not prev_is_bullish:
                        # BEARISH -> BULLISH: Create DEMAND zone from previous bearish period
                        # Condition: aggClose > prevDemandHigh (close broke above demand zone)
                        if (not demand_zone_used and prev_high is not None and 
                            agg_close > prev_high):
                            
                            demand_zones.append({
                                'start_idx': prev_start_bar,
                                'end_idx': prev_start_bar + self.zone_length,
                                'high': prev_high,
                                'low': prev_low,
                                'is_mitigated': False,
                            })
                            demand_zone_used = True
                        
                        # Reset supply state
                        supply_zone_used = False

                    elif not is_bullish and prev_is_bullish:
                        # BULLISH -> BEARISH: Create SUPPLY zone from previous bullish period
                        # Condition: aggClose < prevSupplyLow (close broke below supply zone)
                        if (not supply_zone_used and prev_low is not None and 
                            agg_close < prev_low):
                            
                            supply_zones.append({
                                'start_idx': prev_start_bar,
                                'end_idx': prev_start_bar + self.zone_length,
                                'high': prev_high,
                                'low': prev_low,
                                'is_mitigated': False,
                            })
                            supply_zone_used = True
                        
                        # Reset demand state
                        demand_zone_used = False

                # Check if any zones are mitigated (touched by current aggregated candle)
                for zone in supply_zones:
                    if not zone['is_mitigated']:
                        # Supply zone is mitigated if price high >= zone bottom
                        if agg_high >= zone['low']:
                            zone['is_mitigated'] = True
                
                for zone in demand_zones:
                    if not zone['is_mitigated']:
                        # Demand zone is mitigated if price low <= zone top
                        if agg_low <= zone['high']:
                            zone['is_mitigated'] = True

                # Store current synthetic candle for next iteration
                prev_low = agg_low
                prev_high = agg_high
                prev_start_bar = group_start_idx
                prev_is_bullish = is_bullish
                
                # Start new aggregation group
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]

        return {'supply': supply_zones, 'demand': demand_zones}


class ModelEnsembleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Crypto Prediction System + Supply/Demand Zones')
        self.root.geometry('1400x850')
        
        self.df = None
        self.models = {}
        
        # Create tab framework
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Data Loading
        self.load_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.load_frame, text='Data Loading')
        self.setup_load_tab()
        
        # Tab 2: Feature Engineering
        self.feature_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.feature_frame, text='Feature Engineering')
        self.setup_feature_tab()
        
        # Tab 3: Model Training
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text='Model Training')
        self.setup_train_tab()
        
        # Tab 4: Model Evaluation
        self.eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.eval_frame, text='Model Evaluation')
        self.setup_eval_tab()
        
        # Tab 5: Prediction
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text='Prediction')
        self.setup_predict_tab()
        
        # Tab 6: Supply/Demand Zones
        self.supply_demand_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.supply_demand_frame, text='Supply/Demand Zones')
        self.setup_supply_demand_tab()

    def setup_load_tab(self):
        frame = ttk.LabelFrame(self.load_frame, text='Data Loading', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Button(frame, text='Load Local CSV/Parquet', 
                  command=self.load_local_data).pack(pady=10)
        
        ttk.Button(frame, text='Load Default Data (data/btc_15m.parquet)', 
                  command=self.load_default_data).pack(pady=10)
        
        self.load_status = ttk.Label(frame, text='No data loaded', foreground='red')
        self.load_status.pack(pady=10)
        
        # Data statistics
        self.load_info = ttk.Label(frame, text='', justify=tk.LEFT)
        self.load_info.pack(pady=10)

    def setup_feature_tab(self):
        frame = ttk.LabelFrame(self.feature_frame, text='Feature Engineering', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text='Feature Development In Progress...').pack(pady=10)
        ttk.Label(frame, text='This tab will be used for feature engineering configuration and data preprocessing').pack()

    def setup_train_tab(self):
        frame = ttk.LabelFrame(self.train_frame, text='Model Training', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text='Model Development In Progress...').pack(pady=10)
        ttk.Label(frame, text='This tab will be used for training LightGBM/XGBoost models').pack()

    def setup_eval_tab(self):
        frame = ttk.LabelFrame(self.eval_frame, text='Model Evaluation', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text='Evaluation Development In Progress...').pack(pady=10)
        ttk.Label(frame, text='This tab will be used for model performance evaluation').pack()

    def setup_predict_tab(self):
        frame = ttk.LabelFrame(self.predict_frame, text='Prediction', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text='Prediction Development In Progress...').pack(pady=10)
        ttk.Label(frame, text='This tab will be used for model predictions').pack()

    def setup_supply_demand_tab(self):
        frame = ttk.Frame(self.supply_demand_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Parameter frame
        param_frame = ttk.LabelFrame(frame, text='Detection Parameters', padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text='Aggregation Factor:').pack(side=tk.LEFT, padx=5)
        self.agg_spinbox = ttk.Spinbox(param_frame, from_=1, to=50, width=10)
        self.agg_spinbox.set(4)
        self.agg_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(param_frame, text='Zone Length:').pack(side=tk.LEFT, padx=5)
        self.zone_length_spinbox = ttk.Spinbox(param_frame, from_=10, to=500, width=10)
        self.zone_length_spinbox.set(50)
        self.zone_length_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(param_frame, text='Detect and Plot', 
                  command=self.detect_and_plot).pack(side=tk.LEFT, padx=5)
        
        # Legend
        legend_frame = ttk.LabelFrame(frame, text='Zone Status Legend', padding=10)
        legend_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(legend_frame, text='Red = Supply Zone | Green = Demand Zone | Blue = Mitigated (Touched)', 
                 foreground='gray').pack()
        
        # Chart frame
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
                messagebox.showinfo('Success', f'Loaded {len(self.df)} rows of data')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load data: {str(e)}')

    def load_default_data(self):
        try:
            path = 'data/btc_15m.parquet'
            if not Path(path).exists():
                path = 'btc_15m.parquet'
            
            self.df = pd.read_parquet(path)
            self.update_load_status()
            messagebox.showinfo('Success', f'Loaded default data: {len(self.df)} rows')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load default data: {str(e)}')

    def update_load_status(self):
        if self.df is not None:
            self.load_status.config(text=f'Data loaded: {len(self.df)} rows', foreground='green')
            info_text = f"""Data Information:
Rows: {len(self.df)}
Columns: {', '.join(self.df.columns[:5])}...
Memory: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"""
            self.load_info.config(text=info_text)

    def detect_and_plot(self):
        if self.df is None:
            messagebox.showwarning('Warning', 'Please load data first')
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
            
            # Count mitigated zones
            supply_mitigated = sum(1 for z in zones['supply'] if z['is_mitigated'])
            demand_mitigated = sum(1 for z in zones['demand'] if z['is_mitigated'])
            
            messagebox.showinfo('Detection Complete', 
                f'Supply Zones: {len(zones["supply"])} ({supply_mitigated} mitigated)\n'
                f'Demand Zones: {len(zones["demand"])} ({demand_mitigated} mitigated)')
        except Exception as e:
            messagebox.showerror('Error', f'Detection failed: {str(e)}')

    def plot_kline_with_zones(self, zones, agg_factor):
        display_bars = 1000
        df = self.df.iloc[-display_bars:].reset_index(drop=True) if len(self.df) > display_bars else self.df.reset_index(drop=True)
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Get price range for proper scaling
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        # Calculate offset for zone indexing (zones are in original dataframe indices)
        offset = len(self.df) - len(df)
        
        # Plot Supply Zones
        for zone in zones['supply']:
            # Convert zone indices to display window indices
            start_idx = zone['start_idx'] - offset
            end_idx = zone['end_idx'] - offset
            
            # Only draw if zone is visible in current display window
            if end_idx >= 0 and start_idx < len(df):
                start_idx = max(0, start_idx)
                end_idx = min(len(df), end_idx)
                
                # Determine color based on mitigation status
                if zone['is_mitigated']:
                    color = 'blue'
                    alpha = 0.3
                    label = 'Mitigated Supply'
                else:
                    color = 'red'
                    alpha = 0.2
                    label = 'Active Supply'
                
                rect = mpatches.Rectangle((start_idx - 0.4, zone['low']), 
                                         end_idx - start_idx + 0.8, 
                                         zone['high'] - zone['low'],
                                         linewidth=1.5, edgecolor=color, 
                                         facecolor=color, alpha=alpha)
                ax.add_patch(rect)
        
        # Plot Demand Zones
        for zone in zones['demand']:
            # Convert zone indices to display window indices
            start_idx = zone['start_idx'] - offset
            end_idx = zone['end_idx'] - offset
            
            # Only draw if zone is visible in current display window
            if end_idx >= 0 and start_idx < len(df):
                start_idx = max(0, start_idx)
                end_idx = min(len(df), end_idx)
                
                # Determine color based on mitigation status
                if zone['is_mitigated']:
                    color = 'blue'
                    alpha = 0.3
                    label = 'Mitigated Demand'
                else:
                    color = 'green'
                    alpha = 0.2
                    label = 'Active Demand'
                
                rect = mpatches.Rectangle((start_idx - 0.4, zone['low']), 
                                         end_idx - start_idx + 0.8, 
                                         zone['high'] - zone['low'],
                                         linewidth=1.5, edgecolor=color, 
                                         facecolor=color, alpha=alpha)
                ax.add_patch(rect)
        
        # Plot K-lines on top
        width = 0.6
        for i in range(len(df)):
            o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
            color = 'green' if c >= o else 'red'
            
            # High-Low line (wick)
            ax.plot([i, i], [l, h], color=color, linewidth=0.8)
            
            # Open-Close rectangle (body)
            body_height = abs(c - o) if abs(c - o) > 0 else price_range * 0.001
            body_bottom = min(o, c)
            ax.bar(i, body_height, width=width, bottom=body_bottom, 
                   color=color, alpha=0.8, edgecolor=color, linewidth=0.5)
        
        # Format axes
        supply_active = sum(1 for z in zones['supply'] if not z['is_mitigated'])
        supply_mitigated = sum(1 for z in zones['supply'] if z['is_mitigated'])
        demand_active = sum(1 for z in zones['demand'] if not z['is_mitigated'])
        demand_mitigated = sum(1 for z in zones['demand'] if z['is_mitigated'])
        
        ax.set_xlabel(f'Bar Index (Last {min(len(df), display_bars)} bars)')
        ax.set_ylabel('Price (USDT)')
        ax.set_title(f'K-line Chart with Supply/Demand Zones (Agg: {agg_factor}, Supply: {supply_active}|{supply_mitigated}, Demand: {demand_active}|{demand_mitigated})')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, len(df))
        ax.set_ylim(price_min - price_range * 0.05, price_max + price_range * 0.05)
        
        self.fig.tight_layout()
        self.chart_canvas.draw()


def main():
    root = tk.Tk()
    app = ModelEnsembleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
