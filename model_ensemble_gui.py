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


class SmartMoneyStructure:
    """Smart Money Concepts - Structure Detection (BOS/CHoCH, Order Blocks)"""
    def __init__(self, swing_length=50, internal_length=5):
        self.swing_length = swing_length
        self.internal_length = internal_length

    def detect_swings(self, df, size=50):
        """
        Detect swing points (HH, HL, LL, LH) using lookback window
        Returns: list of swings with type, price, index, time
        """
        h = df['high'].values
        l = df['low'].values
        n = len(df)
        
        if n < size:
            return []
        
        swings = []
        trend = None  # 'up' for bullish leg (making lows), 'down' for bearish leg (making highs)
        
        last_swing_price = None
        last_swing_idx = None
        last_swing_is_high = None
        
        for i in range(size, n):
            # Look back window
            window_high = np.max(h[i-size:i])
            window_low = np.min(l[i-size:i])
            
            # New high in window = bearish leg confirmed
            if h[i] == window_high and trend != 'down':
                if trend == 'up' and last_swing_is_high is False:
                    # Confirm the low
                    trend = 'down'
                    
                    # Find exact index of that low
                    low_idx = np.argmin(l[max(0, last_swing_idx-size):i]) + max(0, last_swing_idx-size)
                    low_price = l[low_idx]
                    
                    swing_type = 'LL' if last_swing_price is None or low_price < last_swing_price else 'HL'
                    swings.append({
                        'type': swing_type,
                        'price': low_price,
                        'index': low_idx,
                        'is_high': False,
                    })
                    last_swing_price = low_price
                    last_swing_idx = low_idx
                    last_swing_is_high = False
                
                trend = 'down'
            
            # New low in window = bullish leg confirmed
            elif l[i] == window_low and trend != 'up':
                if trend == 'down' and last_swing_is_high is True:
                    # Confirm the high
                    trend = 'up'
                    
                    # Find exact index of that high
                    high_idx = np.argmax(h[max(0, last_swing_idx-size):i]) + max(0, last_swing_idx-size)
                    high_price = h[high_idx]
                    
                    swing_type = 'HH' if last_swing_price is None or high_price > last_swing_price else 'LH'
                    swings.append({
                        'type': swing_type,
                        'price': high_price,
                        'index': high_idx,
                        'is_high': True,
                    })
                    last_swing_price = high_price
                    last_swing_idx = high_idx
                    last_swing_is_high = True
                
                trend = 'up'
        
        return swings

    def detect_structures(self, df, swings):
        """
        Detect BOS (Break of Structure) and CHoCH (Change of Character)
        BOS: Price breaks previous swing without changing direction
        CHoCH: Price breaks previous swing AND changes direction
        """
        c = df['close'].values
        structures = []
        
        if len(swings) < 3:
            return structures
        
        # Check structures between consecutive swing reversals
        for i in range(1, len(swings)):
            curr = swings[i]
            prev = swings[i-1]
            
            # Current is high
            if curr['is_high']:
                # Previous should be low, compare current high with high before that
                if i >= 2:
                    prev_swing_of_same_type = swings[i-2]
                    if prev_swing_of_same_type['is_high']:
                        if curr['price'] > prev_swing_of_same_type['price']:
                            structures.append({
                                'type': 'CHoCH',
                                'direction': 'bullish',
                                'price': curr['price'],
                                'index': curr['index'],
                                'label': f"CHoCH (H: {curr['price']:.2f})",
                            })
                        else:
                            structures.append({
                                'type': 'BOS',
                                'direction': 'bearish',
                                'price': curr['price'],
                                'index': curr['index'],
                                'label': f"BOS (H: {curr['price']:.2f})",
                            })
            # Current is low
            else:
                # Previous should be high, compare current low with low before that
                if i >= 2:
                    prev_swing_of_same_type = swings[i-2]
                    if not prev_swing_of_same_type['is_high']:
                        if curr['price'] < prev_swing_of_same_type['price']:
                            structures.append({
                                'type': 'CHoCH',
                                'direction': 'bearish',
                                'price': curr['price'],
                                'index': curr['index'],
                                'label': f"CHoCH (L: {curr['price']:.2f})",
                            })
                        else:
                            structures.append({
                                'type': 'BOS',
                                'direction': 'bullish',
                                'price': curr['price'],
                                'index': curr['index'],
                                'label': f"BOS (L: {curr['price']:.2f})",
                            })
        
        return structures

    def detect_order_blocks(self, df, swings):
        """
        Detect Order Blocks at swing reversals
        OB is the range from pivot to pivot in the direction of reversal
        """
        h = df['high'].values
        l = df['low'].values
        
        order_blocks = []
        
        for i in range(1, len(swings) - 1):
            curr = swings[i]
            next_swing = swings[i + 1]
            
            # When reversing from high to low: create bearish OB
            if curr['is_high'] and not next_swing['is_high']:
                # Range between these two swings
                start_idx = curr['index']
                end_idx = next_swing['index']
                
                if start_idx < end_idx:
                    segment_high = np.max(h[start_idx:end_idx+1])
                    segment_low = np.min(l[start_idx:end_idx+1])
                    
                    order_blocks.append({
                        'type': 'bearish',
                        'high': segment_high,
                        'low': segment_low,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'is_mitigated': False,
                    })
            
            # When reversing from low to high: create bullish OB
            elif not curr['is_high'] and next_swing['is_high']:
                # Range between these two swings
                start_idx = curr['index']
                end_idx = next_swing['index']
                
                if start_idx < end_idx:
                    segment_high = np.max(h[start_idx:end_idx+1])
                    segment_low = np.min(l[start_idx:end_idx+1])
                    
                    order_blocks.append({
                        'type': 'bullish',
                        'high': segment_high,
                        'low': segment_low,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'is_mitigated': False,
                    })
        
        return order_blocks

    def track_mitigation(self, df, order_blocks):
        """
        Track if order blocks are mitigated (touched by current candle)
        """
        h = df['high'].values
        l = df['low'].values
        n = len(df)
        
        for ob in order_blocks:
            for i in range(ob['end_idx'], n):
                if ob['type'] == 'bullish':
                    # Bullish OB mitigated when price touches or goes below the low
                    if l[i] <= ob['low']:
                        ob['is_mitigated'] = True
                        ob['mitigated_idx'] = i
                        break
                else:  # bearish
                    # Bearish OB mitigated when price touches or goes above the high
                    if h[i] >= ob['high']:
                        ob['is_mitigated'] = True
                        ob['mitigated_idx'] = i
                        break
        
        return order_blocks

    def analyze(self, df):
        """Complete SMC analysis"""
        swings = self.detect_swings(df, self.swing_length)
        structures = self.detect_structures(df, swings)
        order_blocks = self.detect_order_blocks(df, swings)
        order_blocks = self.track_mitigation(df, order_blocks)
        
        return {
            'swings': swings,
            'structures': structures,
            'order_blocks': order_blocks,
        }


class ModelEnsembleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Crypto Prediction System + Smart Money Concepts')
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
        
        # Tab 6: Smart Money Concepts
        self.smc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.smc_frame, text='Smart Money Concepts')
        self.setup_smc_tab()

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

    def setup_smc_tab(self):
        frame = ttk.Frame(self.smc_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Parameter frame
        param_frame = ttk.LabelFrame(frame, text='SMC Detection Parameters', padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text='Swing Length:').pack(side=tk.LEFT, padx=5)
        self.swing_length_spinbox = ttk.Spinbox(param_frame, from_=10, to=200, width=10)
        self.swing_length_spinbox.set(50)
        self.swing_length_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(param_frame, text='Analyze SMC', 
                  command=self.analyze_smc).pack(side=tk.LEFT, padx=5)
        
        # Legend
        legend_frame = ttk.LabelFrame(frame, text='Legend', padding=10)
        legend_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(legend_frame, text='HH=Higher High | HL=Higher Low | LL=Lower Low | LH=Lower High', 
                 foreground='gray').pack()
        ttk.Label(legend_frame, text='Red Box=Bearish OB | Green Box=Bullish OB | Blue=Mitigated', 
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

    def analyze_smc(self):
        if self.df is None:
            messagebox.showwarning('Warning', 'Please load data first')
            return
        
        try:
            swing_length = int(self.swing_length_spinbox.get())
            
            smc = SmartMoneyStructure(swing_length=swing_length)
            result = smc.analyze(self.df)
            self.plot_smc_analysis(result, swing_length)
            
            messagebox.showinfo('Analysis Complete', 
                f'Swings: {len(result["swings"])}\n'
                f'Structures: {len(result["structures"])}\n'
                f'Order Blocks: {len(result["order_blocks"])}')
        except Exception as e:
            messagebox.showerror('Error', f'Analysis failed: {str(e)}')

    def plot_smc_analysis(self, result, swing_length):
        display_bars = 1000
        df = self.df.iloc[-display_bars:].reset_index(drop=True) if len(self.df) > display_bars else self.df.reset_index(drop=True)
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Get price range
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        # Calculate offset
        offset = len(self.df) - len(df)
        
        # Plot Order Blocks
        for ob in result['order_blocks']:
            start_idx = ob['start_idx'] - offset
            end_idx = ob['end_idx'] - offset
            
            if end_idx >= 0 and start_idx < len(df):
                start_idx = max(0, start_idx)
                end_idx = min(len(df), end_idx)
                
                color = 'blue' if ob['is_mitigated'] else ('red' if ob['type'] == 'bearish' else 'green')
                alpha = 0.4 if ob['is_mitigated'] else 0.2
                
                rect = mpatches.Rectangle((start_idx - 0.4, ob['low']), 
                                         end_idx - start_idx + 0.8, 
                                         ob['high'] - ob['low'],
                                         linewidth=1.5, edgecolor=color, 
                                         facecolor=color, alpha=alpha)
                ax.add_patch(rect)
        
        # Plot K-lines
        width = 0.6
        for i in range(len(df)):
            o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
            color = 'green' if c >= o else 'red'
            
            ax.plot([i, i], [l, h], color=color, linewidth=0.8)
            body_height = abs(c - o) if abs(c - o) > 0 else price_range * 0.001
            body_bottom = min(o, c)
            ax.bar(i, body_height, width=width, bottom=body_bottom, 
                   color=color, alpha=0.8, edgecolor=color, linewidth=0.5)
        
        # Plot swing points
        for swing in result['swings']:
            idx = swing['index'] - offset
            if 0 <= idx < len(df):
                marker = '^' if swing['is_high'] else 'v'
                color = 'darkred' if swing['is_high'] else 'darkgreen'
                ax.plot(idx, swing['price'], marker=marker, color=color, markersize=8, zorder=5)
                ax.text(idx, swing['price'], f" {swing['type']}", fontsize=7, ha='left')
        
        # Format
        ax.set_xlabel(f'Bar Index (Last {min(len(df), display_bars)} bars)')
        ax.set_ylabel('Price (USDT)')
        ax.set_title(f'Smart Money Concepts (Swing: {swing_length})')
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
