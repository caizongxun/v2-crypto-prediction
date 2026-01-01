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
from pathlib import Path

# Set matplotlib font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100


class SmartMoneyStructure:
    """Smart Money Concepts - Structure Detection matching Pine Script logic"""
    def __init__(self, swing_length=50, internal_length=5):
        self.swing_length = swing_length
        self.internal_length = internal_length
        
        # Constants
        self.BULLISH_LEG = 1
        self.BEARISH_LEG = 0
        self.BULLISH = 1
        self.BEARISH = -1

    def get_leg(self, df, size=50):
        """
        Get current leg: 0 = bearish (making new highs), 1 = bullish (making new lows)
        Matches Pine Script: leg(int size) function
        """
        h = df['high'].values
        l = df['low'].values
        n = len(df)
        
        leg = np.zeros(n)
        
        for i in range(size, n):
            # Get highest in last 'size' bars
            window_high = np.max(h[i-size:i])
            window_low = np.min(l[i-size:i])
            
            if i > 0:
                leg[i] = leg[i-1]
            
            # New high = bearish leg
            if h[i] > window_high:
                leg[i] = self.BEARISH_LEG
            # New low = bullish leg  
            elif l[i] < window_low:
                leg[i] = self.BULLISH_LEG
        
        return leg

    def detect_pivots(self, df, size=50):
        """
        Detect pivot points (HH, HL, LL, LH) using leg changes
        Returns: dict with 'high' and 'low' pivot sequences
        """
        leg = self.get_leg(df, size)
        h = df['high'].values
        l = df['low'].values
        n = len(df)
        
        pivots = {'high': [], 'low': []}
        
        last_high = None
        last_low = None
        
        for i in range(1, n):
            # Leg changed
            if leg[i] != leg[i-1]:
                # From bearish to bullish: create pivot low
                if leg[i-1] == self.BEARISH_LEG and leg[i] == self.BULLISH_LEG:
                    # Find the lowest low in the bearish segment
                    segment_start = max(0, i - size)
                    lowest_idx = np.argmin(l[segment_start:i]) + segment_start
                    lowest_price = l[lowest_idx]
                    
                    pivot_type = 'LL' if last_low is None or lowest_price < last_low else 'HL'
                    pivots['low'].append({
                        'type': pivot_type,
                        'price': lowest_price,
                        'index': lowest_idx,
                        'bar': i-1
                    })
                    last_low = lowest_price
                
                # From bullish to bearish: create pivot high
                elif leg[i-1] == self.BULLISH_LEG and leg[i] == self.BEARISH_LEG:
                    # Find the highest high in the bullish segment
                    segment_start = max(0, i - size)
                    highest_idx = np.argmax(h[segment_start:i]) + segment_start
                    highest_price = h[highest_idx]
                    
                    pivot_type = 'HH' if last_high is None or highest_price > last_high else 'LH'
                    pivots['high'].append({
                        'type': pivot_type,
                        'price': highest_price,
                        'index': highest_idx,
                        'bar': i-1
                    })
                    last_high = highest_price
        
        return pivots, leg

    def detect_structures(self, df, pivots, leg):
        """
        Detect BOS and CHoCH using crossover/crossunder logic
        Matches Pine Script: ta.crossover(close, pivot.level)
        """
        c = df['close'].values
        n = len(df)
        
        structures = []
        
        # Track current pivot levels
        current_high_pivot = None
        current_low_pivot = None
        
        high_pivot_idx = 0
        low_pivot_idx = 0
        
        high_crossed = False
        low_crossed = False
        
        for i in range(1, n):
            # Update pivots based on sequence
            if high_pivot_idx < len(pivots['high']):
                next_high = pivots['high'][high_pivot_idx]
                if i == next_high['index'] + 1:  # New pivot detected
                    current_high_pivot = next_high
                    high_crossed = False
                    high_pivot_idx += 1
            
            if low_pivot_idx < len(pivots['low']):
                next_low = pivots['low'][low_pivot_idx]
                if i == next_low['index'] + 1:  # New pivot detected
                    current_low_pivot = next_low
                    low_crossed = False
                    low_pivot_idx += 1
            
            # Check for crossovers at high pivot
            if current_high_pivot is not None and not high_crossed:
                # Bullish structure: price crosses above high pivot
                if c[i-1] <= current_high_pivot['price'] < c[i]:
                    # Determine if BOS or CHoCH
                    structure_type = 'CHoCH' if leg[i] == self.BULLISH_LEG else 'BOS'
                    structures.append({
                        'type': structure_type,
                        'direction': 'bullish',
                        'price': current_high_pivot['price'],
                        'index': i,
                        'pivot_index': current_high_pivot['index'],
                    })
                    high_crossed = True
            
            # Check for crossunders at low pivot
            if current_low_pivot is not None and not low_crossed:
                # Bearish structure: price crosses below low pivot
                if c[i-1] >= current_low_pivot['price'] > c[i]:
                    # Determine if BOS or CHoCH
                    structure_type = 'CHoCH' if leg[i] == self.BEARISH_LEG else 'BOS'
                    structures.append({
                        'type': structure_type,
                        'direction': 'bearish',
                        'price': current_low_pivot['price'],
                        'index': i,
                        'pivot_index': current_low_pivot['index'],
                    })
                    low_crossed = True
        
        return structures

    def detect_order_blocks(self, df, pivots, leg):
        """
        Detect Order Blocks at pivot reversals
        Matches Pine Script: storeOrdeBlock function logic
        """
        h = df['high'].values
        l = df['low'].values
        
        order_blocks = []
        
        # Get all pivots in chronological order
        all_pivots = []
        for p in pivots['high']:
            all_pivots.append(('high', p))
        for p in pivots['low']:
            all_pivots.append(('low', p))
        all_pivots.sort(key=lambda x: x[1]['index'])
        
        # Create OB at each transition
        for i in range(len(all_pivots) - 1):
            curr_type, curr_pivot = all_pivots[i]
            next_type, next_pivot = all_pivots[i+1]
            
            start_idx = curr_pivot['index']
            end_idx = next_pivot['index']
            
            if start_idx < end_idx:
                segment_h = h[start_idx:end_idx+1]
                segment_l = l[start_idx:end_idx+1]
                
                if len(segment_h) > 0 and len(segment_l) > 0:
                    # Bearish OB: high to low transition
                    if curr_type == 'high' and next_type == 'low':
                        ob_high = np.max(segment_h)
                        ob_low = np.min(segment_l)
                        ob_idx = np.argmax(segment_h) + start_idx
                        
                        order_blocks.append({
                            'type': 'bearish',
                            'high': ob_high,
                            'low': ob_low,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'block_idx': ob_idx,
                            'is_mitigated': False,
                        })
                    
                    # Bullish OB: low to high transition
                    elif curr_type == 'low' and next_type == 'high':
                        ob_high = np.max(segment_h)
                        ob_low = np.min(segment_l)
                        ob_idx = np.argmin(segment_l) + start_idx
                        
                        order_blocks.append({
                            'type': 'bullish',
                            'high': ob_high,
                            'low': ob_low,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'block_idx': ob_idx,
                            'is_mitigated': False,
                        })
        
        return order_blocks

    def track_mitigation(self, df, order_blocks):
        """
        Track OB mitigation (crossing current price)
        """
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)
        
        for ob in order_blocks:
            for i in range(ob['end_idx'], n):
                if ob['type'] == 'bullish':
                    # Bullish OB mitigated when price closes below low
                    if l[i] <= ob['low']:
                        ob['is_mitigated'] = True
                        ob['mitigated_idx'] = i
                        break
                else:  # bearish
                    # Bearish OB mitigated when price closes above high
                    if h[i] >= ob['high']:
                        ob['is_mitigated'] = True
                        ob['mitigated_idx'] = i
                        break
        
        return order_blocks

    def analyze(self, df):
        """Complete SMC analysis"""
        pivots, leg = self.detect_pivots(df, self.swing_length)
        structures = self.detect_structures(df, pivots, leg)
        order_blocks = self.detect_order_blocks(df, pivots, leg)
        order_blocks = self.track_mitigation(df, order_blocks)
        
        return {
            'pivots': pivots,
            'structures': structures,
            'order_blocks': order_blocks,
            'leg': leg,
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
        
        self.load_info = ttk.Label(frame, text='', justify=tk.LEFT)
        self.load_info.pack(pady=10)

    def setup_feature_tab(self):
        frame = ttk.LabelFrame(self.feature_frame, text='Feature Engineering', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(frame, text='Feature Development In Progress...').pack(pady=10)

    def setup_train_tab(self):
        frame = ttk.LabelFrame(self.train_frame, text='Model Training', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(frame, text='Model Development In Progress...').pack(pady=10)

    def setup_eval_tab(self):
        frame = ttk.LabelFrame(self.eval_frame, text='Model Evaluation', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(frame, text='Evaluation Development In Progress...').pack(pady=10)

    def setup_predict_tab(self):
        frame = ttk.LabelFrame(self.predict_frame, text='Prediction', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(frame, text='Prediction Development In Progress...').pack(pady=10)

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
        ttk.Label(legend_frame, text='Blue Box=Bearish OB | Green Box=Bullish OB | Pink=Mitigated', 
                 foreground='gray').pack()
        ttk.Label(legend_frame, text='Yellow Line=BOS | Cyan Line=CHoCH', 
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
                filetypes=[("CSV files", "*.csv"), ("Parquet files", "*.parquet")]
            )
            if filepath:
                if filepath.endswith('.csv'):
                    self.df = pd.read_csv(filepath)
                else:
                    self.df = pd.read_parquet(filepath)
                self.update_load_status()
                messagebox.showinfo('Success', f'Loaded {len(self.df)} rows')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load: {str(e)}')

    def load_default_data(self):
        try:
            path = 'data/btc_15m.parquet'
            if not Path(path).exists():
                path = 'btc_15m.parquet'
            self.df = pd.read_parquet(path)
            self.update_load_status()
            messagebox.showinfo('Success', f'Loaded {len(self.df)} rows')
        except Exception as e:
            messagebox.showerror('Error', f'Failed: {str(e)}')

    def update_load_status(self):
        if self.df is not None:
            self.load_status.config(text=f'Loaded: {len(self.df)} rows', foreground='green')
            info = f"Rows: {len(self.df)}\nColumns: {', '.join(self.df.columns[:5])}"
            self.load_info.config(text=info)

    def analyze_smc(self):
        if self.df is None:
            messagebox.showwarning('Warning', 'Please load data first')
            return
        
        try:
            swing_length = int(self.swing_length_spinbox.get())
            smc = SmartMoneyStructure(swing_length=swing_length)
            result = smc.analyze(self.df)
            self.plot_smc_analysis(result, swing_length)
            
            msg = (f"Swing Pivots: {len(result['pivots']['high']) + len(result['pivots']['low'])}\n"
                   f"Structures: {len(result['structures'])}\n"
                   f"Order Blocks: {len(result['order_blocks'])}")
            messagebox.showinfo('Complete', msg)
        except Exception as e:
            messagebox.showerror('Error', f'{str(e)}')
            import traceback
            traceback.print_exc()

    def plot_smc_analysis(self, result, swing_length):
        display_bars = min(1000, len(self.df))
        df = self.df.iloc[-display_bars:].reset_index(drop=True)
        
        # Calculate offset: original index = display index + offset
        offset = len(self.df) - display_bars
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        # Plot Order Blocks as rectangles aligned to bar positions
        for ob in result['order_blocks']:
            start_bar = ob['start_idx'] - offset  # Convert to display index
            end_bar = ob['end_idx'] - offset
            
            # Only plot if visible in current display
            if end_bar >= 0 and start_bar < len(df):
                # Clamp to display range
                plot_start = max(0, start_bar)
                plot_end = min(len(df) - 1, end_bar)
                
                # Create rectangle from start bar to end bar
                rect_width = plot_end - plot_start + 1
                
                color = 'blue' if ob['is_mitigated'] else (
                    'red' if ob['type'] == 'bearish' else 'green'
                )
                alpha = 0.3 if ob['is_mitigated'] else 0.15
                
                # Rectangle with x-axis aligned to bar indices
                rect = mpatches.Rectangle(
                    (plot_start - 0.4, ob['low']),  # Start position
                    rect_width + 0.8,  # Width in bar units
                    ob['high'] - ob['low'],  # Height in price
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha,
                    zorder=1
                )
                ax.add_patch(rect)
        
        # Plot K-lines (candlesticks)
        for i in range(len(df)):
            o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
            color = 'green' if c >= o else 'red'
            
            # Wick (high-low line)
            ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)
            
            # Body (open-close rectangle)
            body = abs(c - o) if abs(c - o) > 0 else price_range * 0.001
            body_bottom = min(o, c)
            ax.bar(i, body, width=0.6, bottom=body_bottom, 
                   color=color, alpha=0.9, edgecolor=color, linewidth=0.5, zorder=2)
        
        # Plot pivot points with labels
        for p in result['pivots']['high']:
            idx = p['index'] - offset
            if 0 <= idx < len(df):
                ax.plot(idx, p['price'], marker='^', color='darkred', markersize=8, zorder=5)
                ax.text(idx, p['price'] + price_range * 0.02, p['type'], 
                       fontsize=7, ha='center', color='darkred', fontweight='bold')
        
        for p in result['pivots']['low']:
            idx = p['index'] - offset
            if 0 <= idx < len(df):
                ax.plot(idx, p['price'], marker='v', color='darkgreen', markersize=8, zorder=5)
                ax.text(idx, p['price'] - price_range * 0.02, p['type'], 
                       fontsize=7, ha='center', color='darkgreen', fontweight='bold')
        
        # Plot structures (BOS/CHoCH) as vertical lines at exact bar positions
        for struct in result['structures']:
            idx = struct['index'] - offset
            if 0 <= idx < len(df):
                # Vertical line at structure cross point
                color = 'cyan' if struct['type'] == 'CHoCH' else 'gold'
                linewidth = 2 if struct['type'] == 'CHoCH' else 1.5
                linestyle = '-' if struct['type'] == 'CHoCH' else '--'
                
                ax.axvline(x=idx, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.7, zorder=3)
                
                # Label at top
                label_y = price_max + price_range * 0.01
                ax.text(idx, label_y, struct['type'], fontsize=7, ha='center', 
                       color=color, fontweight='bold', rotation=0, zorder=4)
        
        # Formatting
        ax.set_xlabel(f'Bar Index (Last {display_bars} bars)', fontsize=10)
        ax.set_ylabel('Price (USDT)', fontsize=10)
        ax.set_title(f'Smart Money Concepts - Swing Length: {swing_length}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.set_xlim(-1, len(df))
        ax.set_ylim(price_min - price_range * 0.1, price_max + price_range * 0.1)
        
        # Add background color zones for better visibility
        ax.set_facecolor('#f8f9fa')
        
        self.fig.tight_layout()
        self.chart_canvas.draw()


def main():
    root = tk.Tk()
    app = ModelEnsembleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
