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
    """Smart Money Concepts - Corrected Structure Detection"""
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
        Improved logic with better initialization
        """
        h = df['high'].values
        l = df['low'].values
        n = len(df)
        
        leg = np.zeros(n)
        
        # Initialize: find initial trend
        if size < n:
            initial_high = np.max(h[:size])
            initial_low = np.min(l[:size])
            # Start with bearish if we're closer to high, bullish if closer to low
            leg[size] = self.BULLISH_LEG if h[size] > initial_high else self.BEARISH_LEG
        
        for i in range(size + 1, n):
            leg[i] = leg[i - 1]
            
            # Find the highest high and lowest low in the look-back window
            window_h = h[max(0, i - size):i]
            window_l = l[max(0, i - size):i]
            
            highest = np.max(window_h)
            lowest = np.min(window_l)
            
            # New high = switch to bearish leg (potential reversal coming)
            if h[i] > highest:
                leg[i] = self.BEARISH_LEG
            # New low = switch to bullish leg (potential reversal coming)
            elif l[i] < lowest:
                leg[i] = self.BULLISH_LEG
        
        return leg

    def detect_pivots_refined(self, df, size=50):
        """
        Refined pivot detection using leg transitions
        Returns distinct pivot points with better accuracy
        """
        leg = self.get_leg(df, size)
        h = df['high'].values
        l = df['low'].values
        n = len(df)
        
        pivots = {'high': [], 'low': []}
        
        last_high_price = None
        last_low_price = None
        
        for i in range(size + 1, n):
            # Detect leg transition points
            if leg[i] != leg[i - 1]:
                
                # Transitioning from BEARISH to BULLISH
                # This means we've been making lower lows - find the lowest
                if leg[i - 1] == self.BEARISH_LEG and leg[i] == self.BULLISH_LEG:
                    
                    # Find lowest low in the bearish segment
                    search_start = max(size, i - size - 10)
                    segment_l = l[search_start:i]
                    lowest_idx = np.argmin(segment_l) + search_start
                    lowest_price = l[lowest_idx]
                    
                    # Determine pivot type based on comparison with previous low
                    if last_low_price is None:
                        pivot_type = 'LL'
                    elif lowest_price < last_low_price:
                        pivot_type = 'LL'
                    else:
                        pivot_type = 'HL'
                    
                    pivots['low'].append({
                        'type': pivot_type,
                        'price': lowest_price,
                        'index': lowest_idx,
                        'bar_index': i - 1
                    })
                    last_low_price = lowest_price
                
                # Transitioning from BULLISH to BEARISH
                # This means we've been making higher highs - find the highest
                elif leg[i - 1] == self.BULLISH_LEG and leg[i] == self.BEARISH_LEG:
                    
                    # Find highest high in the bullish segment
                    search_start = max(size, i - size - 10)
                    segment_h = h[search_start:i]
                    highest_idx = np.argmax(segment_h) + search_start
                    highest_price = h[highest_idx]
                    
                    # Determine pivot type based on comparison with previous high
                    if last_high_price is None:
                        pivot_type = 'HH'
                    elif highest_price > last_high_price:
                        pivot_type = 'HH'
                    else:
                        pivot_type = 'LH'
                    
                    pivots['high'].append({
                        'type': pivot_type,
                        'price': highest_price,
                        'index': highest_idx,
                        'bar_index': i - 1
                    })
                    last_high_price = highest_price
        
        return pivots, leg

    def detect_structures(self, df, pivots, leg):
        """
        Detect BOS and CHoCH using refined crossover logic
        BOS = Break of Structure (price breaks previous support/resistance)
        CHoCH = Change of Character (leg changes direction)
        """
        c = df['close'].values
        n = len(df)
        
        structures = []
        
        # Merge and sort all pivots by index
        all_pivots = []
        for p in pivots['high']:
            all_pivots.append({
                'type': 'high',
                'pivot_type': p['type'],
                'price': p['price'],
                'index': p['index'],
            })
        for p in pivots['low']:
            all_pivots.append({
                'type': 'low',
                'pivot_type': p['type'],
                'price': p['price'],
                'index': p['index'],
            })
        
        all_pivots.sort(key=lambda x: x['index'])
        
        # Track structure breaks
        for i in range(1, len(all_pivots)):
            curr_pivot = all_pivots[i]
            prev_pivot = all_pivots[i - 1]
            
            if curr_pivot['type'] == 'high' and prev_pivot['type'] == 'low':
                # Looking for bullish break at high pivot level
                pivot_idx = curr_pivot['index']
                pivot_price = curr_pivot['price']
                
                # Check bars after pivot for breakout
                for check_idx in range(pivot_idx + 1, min(pivot_idx + 20, n)):
                    if c[check_idx] > pivot_price and c[check_idx - 1] <= pivot_price:
                        # Determine structure type based on pivot sequence
                        structure_type = 'CHoCH' if curr_pivot['pivot_type'] == 'HH' else 'BOS'
                        
                        structures.append({
                            'type': structure_type,
                            'direction': 'bullish',
                            'price': pivot_price,
                            'index': check_idx,
                            'pivot_index': pivot_idx,
                        })
                        break
            
            elif curr_pivot['type'] == 'low' and prev_pivot['type'] == 'high':
                # Looking for bearish break at low pivot level
                pivot_idx = curr_pivot['index']
                pivot_price = curr_pivot['price']
                
                # Check bars after pivot for breakdown
                for check_idx in range(pivot_idx + 1, min(pivot_idx + 20, n)):
                    if c[check_idx] < pivot_price and c[check_idx - 1] >= pivot_price:
                        # Determine structure type based on pivot sequence
                        structure_type = 'CHoCH' if curr_pivot['pivot_type'] == 'LL' else 'BOS'
                        
                        structures.append({
                            'type': structure_type,
                            'direction': 'bearish',
                            'price': pivot_price,
                            'index': check_idx,
                            'pivot_index': pivot_idx,
                        })
                        break
        
        return structures

    def detect_order_blocks_corrected(self, df, pivots, leg):
        """
        Corrected Order Block detection - strict logic
        
        Bearish OB: Created at the end of a bullish impulse
                    - HH -> LL sequence (clear directional reversal)
                    - OB = the price candles from HH to LL formation
        
        Bullish OB: Created at the end of a bearish impulse
                    - LL -> HH sequence (clear directional reversal)
                    - OB = the price candles from LL to HH formation
        """
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)
        
        order_blocks = []
        
        # Merge all pivots
        all_pivots = []
        for p in pivots['high']:
            all_pivots.append({
                'type': 'high',
                'pivot_type': p['type'],  # HH or LH
                'price': p['price'],
                'index': p['index'],
            })
        for p in pivots['low']:
            all_pivots.append({
                'type': 'low',
                'pivot_type': p['type'],  # LL or HL
                'price': p['price'],
                'index': p['index'],
            })
        
        all_pivots.sort(key=lambda x: x['index'])
        
        # Track significant pivots for directional reversals
        hh_sequence = []  # Sequence of HH pivots
        ll_sequence = []  # Sequence of LL pivots
        
        for pivot in all_pivots:
            if pivot['type'] == 'high':
                if pivot['pivot_type'] == 'HH':
                    hh_sequence.append(pivot)
            else:  # pivot['type'] == 'low'
                if pivot['pivot_type'] == 'LL':
                    ll_sequence.append(pivot)
        
        # Generate bearish OBs: HH -> LL
        for i in range(len(hh_sequence) - 1):
            curr_hh = hh_sequence[i]
            
            # Find the next LL that comes after this HH
            next_ll = None
            for ll in ll_sequence:
                if ll['index'] > curr_hh['index']:
                    next_ll = ll
                    break
            
            if next_ll is not None:
                start_idx = curr_hh['index']
                end_idx = next_ll['index']
                
                if start_idx < end_idx:
                    segment_h = h[start_idx:end_idx + 1]
                    segment_l = l[start_idx:end_idx + 1]
                    
                    ob_high = np.max(segment_h)
                    ob_low = np.min(segment_l)
                    
                    width = end_idx - start_idx
                    height = ob_high - ob_low
                    height_pct = (height / ob_high * 100) if ob_high > 0 else 0
                    
                    # Validation: reasonable size
                    if 5 <= width <= 500 and 0.05 <= height_pct <= 20:
                        order_blocks.append({
                            'type': 'bearish',
                            'high': ob_high,
                            'low': ob_low,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'width': width,
                            'height_pct': height_pct,
                            'is_mitigated': False,
                            'direction': 'HH->LL'
                        })
        
        # Generate bullish OBs: LL -> HH
        for i in range(len(ll_sequence) - 1):
            curr_ll = ll_sequence[i]
            
            # Find the next HH that comes after this LL
            next_hh = None
            for hh in hh_sequence:
                if hh['index'] > curr_ll['index']:
                    next_hh = hh
                    break
            
            if next_hh is not None:
                start_idx = curr_ll['index']
                end_idx = next_hh['index']
                
                if start_idx < end_idx:
                    segment_h = h[start_idx:end_idx + 1]
                    segment_l = l[start_idx:end_idx + 1]
                    
                    ob_high = np.max(segment_h)
                    ob_low = np.min(segment_l)
                    
                    width = end_idx - start_idx
                    height = ob_high - ob_low
                    height_pct = (height / ob_high * 100) if ob_high > 0 else 0
                    
                    # Validation: reasonable size
                    if 5 <= width <= 500 and 0.05 <= height_pct <= 20:
                        order_blocks.append({
                            'type': 'bullish',
                            'high': ob_high,
                            'low': ob_low,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'width': width,
                            'height_pct': height_pct,
                            'is_mitigated': False,
                            'direction': 'LL->HH'
                        })
        
        return order_blocks

    def track_mitigation(self, df, order_blocks):
        """
        Track OB mitigation
        Bearish OB: mitigated when price closes below low
        Bullish OB: mitigated when price closes below low (penetration)
        """
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)
        
        for ob in order_blocks:
            for i in range(ob['end_idx'] + 1, min(ob['end_idx'] + 100, n)):
                if ob['type'] == 'bullish':
                    # Bullish OB mitigated when price closes below low
                    if c[i] < ob['low']:
                        ob['is_mitigated'] = True
                        ob['mitigated_idx'] = i
                        ob['mitigation_price'] = c[i]
                        break
                else:  # bearish
                    # Bearish OB mitigated when price closes above high
                    if c[i] > ob['high']:
                        ob['is_mitigated'] = True
                        ob['mitigated_idx'] = i
                        ob['mitigation_price'] = c[i]
                        break
        
        return order_blocks

    def analyze(self, df):
        """Complete corrected SMC analysis"""
        pivots, leg = self.detect_pivots_refined(df, self.swing_length)
        structures = self.detect_structures(df, pivots, leg)
        order_blocks = self.detect_order_blocks_corrected(df, pivots, leg)
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
        ttk.Label(legend_frame, text='Pivots: HH (Higher High) ↑ | HL (Higher Low) | LL (Lower Low) ↓ | LH (Lower High)', 
                 foreground='gray').pack()
        ttk.Label(legend_frame, text='Order Blocks: Blue Box=Bearish (HH→LL) | Green Box=Bullish (LL→HH)', 
                 foreground='gray').pack()
        ttk.Label(legend_frame, text='Structures: Yellow Line=BOS | Cyan Line=CHoCH', 
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
            
            high_pivots = len(result['pivots']['high'])
            low_pivots = len(result['pivots']['low'])
            structures = len(result['structures'])
            obs = len(result['order_blocks'])
            
            msg = (f"High Pivots: {high_pivots}\n"
                   f"Low Pivots: {low_pivots}\n"
                   f"Structures (BOS/CHoCH): {structures}\n"
                   f"Order Blocks: {obs}")
            messagebox.showinfo('Analysis Complete', msg)
        except Exception as e:
            messagebox.showerror('Error', f'{str(e)}')
            import traceback
            traceback.print_exc()

    def plot_smc_analysis(self, result, swing_length):
        """Enhanced SMC plotting with improved visualization"""
        display_bars = min(1000, len(self.df))
        df = self.df.iloc[-display_bars:].reset_index(drop=True)
        
        # Calculate offset
        offset = len(self.df) - display_bars
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        # Plot Order Blocks with improved rendering
        for ob in result['order_blocks']:
            display_start = ob['start_idx'] - offset
            display_end = ob['end_idx'] - offset
            
            # Clamp to visible range
            if display_end >= 0 and display_start < len(df):
                plot_start = max(0, display_start)
                plot_end = min(len(df) - 1, display_end)
                width = plot_end - plot_start + 1
                
                # Color and alpha based on mitigation status
                if ob['is_mitigated']:
                    if ob['type'] == 'bearish':
                        color = '#FF6B9D'
                        alpha = 0.3
                    else:
                        color = '#FFB3D9'
                        alpha = 0.3
                    linestyle = '--'
                    linewidth = 1.5
                else:
                    if ob['type'] == 'bearish':
                        color = '#4169E1'
                        alpha = 0.25
                    else:
                        color = '#32CD32'
                        alpha = 0.25
                    linestyle = '-'
                    linewidth = 2
                
                # Draw rectangle
                rect = mpatches.Rectangle(
                    (plot_start - 0.5, ob['low']),
                    width,
                    ob['high'] - ob['low'],
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha,
                    linestyle=linestyle,
                    zorder=2
                )
                ax.add_patch(rect)
        
        # Plot candlesticks
        for i in range(len(df)):
            o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
            color = '#00AA00' if c >= o else '#CC0000'
            
            # Wick
            ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=3)
            
            # Body
            body_size = abs(c - o) if abs(c - o) > 0 else price_range * 0.001
            body_bottom = min(o, c)
            ax.bar(i, body_size, width=0.6, bottom=body_bottom,
                   color=color, alpha=0.9, edgecolor=color, linewidth=0.5, zorder=3)
        
        # Plot pivot points - with better positioning
        plotted_highs = set()
        for p in result['pivots']['high']:
            idx = p['index'] - offset
            if 0 <= idx < len(df):
                marker_color = '#8B0000'
                marker = '^'
                offset_pct = 0.025 if idx not in plotted_highs else 0.04
                
                ax.plot(idx, p['price'], marker=marker, color=marker_color, 
                       markersize=9, zorder=5, markeredgewidth=1, markeredgecolor='darkred')
                
                label_y = p['price'] + price_range * offset_pct
                ax.text(idx, label_y, f"{p['type']}", fontsize=7, ha='center', 
                       color=marker_color, fontweight='bold', zorder=6)
                
                plotted_highs.add(idx)
        
        plotted_lows = set()
        for p in result['pivots']['low']:
            idx = p['index'] - offset
            if 0 <= idx < len(df):
                marker_color = '#006400'
                marker = 'v'
                offset_pct = 0.025 if idx not in plotted_lows else 0.04
                
                ax.plot(idx, p['price'], marker=marker, color=marker_color,
                       markersize=9, zorder=5, markeredgewidth=1, markeredgecolor='darkgreen')
                
                label_y = p['price'] - price_range * offset_pct
                ax.text(idx, label_y, f"{p['type']}", fontsize=7, ha='center',
                       color=marker_color, fontweight='bold', zorder=6)
                
                plotted_lows.add(idx)
        
        # Plot structures with cleaner rendering
        for struct in result['structures']:
            idx = struct['index'] - offset
            if 0 <= idx < len(df):
                if struct['type'] == 'CHoCH':
                    color = '#00CED1'
                    linestyle = '-'
                    linewidth = 2
                else:  # BOS
                    color = '#FFD700'
                    linestyle = '--'
                    linewidth = 1.5
                
                ax.axvline(x=idx, color=color, linestyle=linestyle, linewidth=linewidth,
                          alpha=0.6, zorder=4)
        
        # Formatting
        ax.set_xlabel(f'Bar Index (Last {display_bars} bars)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Price (USDT)', fontsize=10, fontweight='bold')
        ax.set_title(f'Smart Money Concepts Analysis - Swing Length: {swing_length}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle=':', color='gray')
        ax.set_xlim(-1, len(df))
        ax.set_ylim(price_min - price_range * 0.1, price_max + price_range * 0.1)
        
        # Background
        ax.set_facecolor('#f8f9fa')
        self.fig.patch.set_facecolor('white')
        
        self.fig.tight_layout()
        self.chart_canvas.draw()


def main():
    root = tk.Tk()
    app = ModelEnsembleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
