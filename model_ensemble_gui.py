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
        Get current leg with improved initialization and logic
        """
        h = df['high'].values
        l = df['low'].values
        n = len(df)
        
        leg = np.zeros(n)
        
        if size < n:
            initial_high = np.max(h[:size])
            initial_low = np.min(l[:size])
            leg[size] = self.BULLISH_LEG if h[size] > initial_high else self.BEARISH_LEG
        
        for i in range(size + 1, n):
            leg[i] = leg[i - 1]
            
            window_h = h[max(0, i - size):i]
            window_l = l[max(0, i - size):i]
            
            highest = np.max(window_h)
            lowest = np.min(window_l)
            
            if h[i] > highest:
                leg[i] = self.BEARISH_LEG
            elif l[i] < lowest:
                leg[i] = self.BULLISH_LEG
        
        return leg

    def detect_pivots(self, df, size=50):
        """Refined pivot detection using leg transitions"""
        leg = self.get_leg(df, size)
        h = df['high'].values
        l = df['low'].values
        n = len(df)
        
        pivots = {'high': [], 'low': []}
        
        last_high_price = None
        last_low_price = None
        
        for i in range(size + 1, n):
            if leg[i] != leg[i - 1]:
                
                # BEARISH to BULLISH
                if leg[i - 1] == self.BEARISH_LEG and leg[i] == self.BULLISH_LEG:
                    
                    search_start = max(size, i - size - 10)
                    segment_l = l[search_start:i]
                    lowest_idx = np.argmin(segment_l) + search_start
                    lowest_price = l[lowest_idx]
                    
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
                
                # BULLISH to BEARISH
                elif leg[i - 1] == self.BULLISH_LEG and leg[i] == self.BEARISH_LEG:
                    
                    search_start = max(size, i - size - 10)
                    segment_h = h[search_start:i]
                    highest_idx = np.argmax(segment_h) + search_start
                    highest_price = h[highest_idx]
                    
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
        """Detect BOS and CHoCH"""
        c = df['close'].values
        n = len(df)
        
        structures = []
        
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
        
        for i in range(1, len(all_pivots)):
            curr_pivot = all_pivots[i]
            prev_pivot = all_pivots[i - 1]
            
            if curr_pivot['type'] == 'high' and prev_pivot['type'] == 'low':
                pivot_idx = curr_pivot['index']
                pivot_price = curr_pivot['price']
                
                for check_idx in range(pivot_idx + 1, min(pivot_idx + 20, n)):
                    if c[check_idx] > pivot_price and c[check_idx - 1] <= pivot_price:
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
                pivot_idx = curr_pivot['index']
                pivot_price = curr_pivot['price']
                
                for check_idx in range(pivot_idx + 1, min(pivot_idx + 20, n)):
                    if c[check_idx] < pivot_price and c[check_idx - 1] >= pivot_price:
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

    def detect_order_blocks(self, df, pivots, leg):
        """Corrected Order Block detection: Bearish OB (HH->LL), Bullish OB (LL->HH)"""
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)
        
        order_blocks = []
        
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
        
        hh_sequence = []
        ll_sequence = []
        
        for pivot in all_pivots:
            if pivot['type'] == 'high' and pivot['pivot_type'] == 'HH':
                hh_sequence.append(pivot)
            elif pivot['type'] == 'low' and pivot['pivot_type'] == 'LL':
                ll_sequence.append(pivot)
        
        # Bearish OBs: HH -> LL
        for curr_hh in hh_sequence:
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
        
        # Bullish OBs: LL -> HH
        for curr_ll in ll_sequence:
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
        """Track OB mitigation status"""
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)
        
        for ob in order_blocks:
            for i in range(ob['end_idx'] + 1, min(ob['end_idx'] + 100, n)):
                if ob['type'] == 'bullish':
                    if c[i] < ob['low']:
                        ob['is_mitigated'] = True
                        ob['mitigated_idx'] = i
                        ob['mitigation_price'] = c[i]
                        break
                else:
                    if c[i] > ob['high']:
                        ob['is_mitigated'] = True
                        ob['mitigated_idx'] = i
                        ob['mitigation_price'] = c[i]
                        break
        
        return order_blocks

    def analyze(self, df):
        """Complete corrected SMC analysis"""
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
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.load_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.load_frame, text='Data Loading')
        self.setup_load_tab()
        
        self.feature_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.feature_frame, text='Feature Engineering')
        self.setup_feature_tab()
        
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text='Model Training')
        self.setup_train_tab()
        
        self.eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.eval_frame, text='Model Evaluation')
        self.setup_eval_tab()
        
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text='Prediction')
        self.setup_predict_tab()
        
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
        
        param_frame = ttk.LabelFrame(frame, text='SMC Detection Parameters', padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text='Swing Length:').pack(side=tk.LEFT, padx=5)
        self.swing_length_spinbox = ttk.Spinbox(param_frame, from_=10, to=200, width=10)
        self.swing_length_spinbox.set(50)
        self.swing_length_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(param_frame, text='Analyze SMC', 
                  command=self.analyze_smc).pack(side=tk.LEFT, padx=5)
        
        legend_frame = ttk.LabelFrame(frame, text='Legend', padding=10)
        legend_frame.pack(fill=tk.X, pady=5)
        ttk.Label(legend_frame, text='Pivots: HH | HL | LL | LH', 
                 foreground='gray').pack()
        ttk.Label(legend_frame, text='OBs: Blue=Bearish(HH->LL) | Green=Bullish(LL->HH)', 
                 foreground='gray').pack()
        ttk.Label(legend_frame, text='Structures: Yellow=BOS | Cyan=CHoCH', 
                 foreground='gray').pack()
        
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
                   f"Structures: {structures}\n"
                   f"Order Blocks: {obs}")
            messagebox.showinfo('Complete', msg)
        except Exception as e:
            messagebox.showerror('Error', f'{str(e)}')
            import traceback
            traceback.print_exc()

    def plot_smc_analysis(self, result, swing_length):
        """Enhanced SMC plotting"""
        display_bars = min(1000, len(self.df))
        df = self.df.iloc[-display_bars:].reset_index(drop=True)
        
        offset = len(self.df) - display_bars
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        # Order Blocks
        for ob in result['order_blocks']:
            display_start = ob['start_idx'] - offset
            display_end = ob['end_idx'] - offset
            
            if display_end >= 0 and display_start < len(df):
                plot_start = max(0, display_start)
                plot_end = min(len(df) - 1, display_end)
                width = plot_end - plot_start + 1
                
                if ob['is_mitigated']:
                    color = '#FF6B9D' if ob['type'] == 'bearish' else '#FFB3D9'
                    alpha = 0.3
                else:
                    color = '#4169E1' if ob['type'] == 'bearish' else '#32CD32'
                    alpha = 0.25
                
                rect = mpatches.Rectangle(
                    (plot_start - 0.5, ob['low']),
                    width,
                    ob['high'] - ob['low'],
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha,
                    zorder=2
                )
                ax.add_patch(rect)
        
        # Candlesticks
        for i in range(len(df)):
            o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
            color = '#00AA00' if c >= o else '#CC0000'
            
            ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=3)
            
            body_size = abs(c - o) if abs(c - o) > 0 else price_range * 0.001
            body_bottom = min(o, c)
            ax.bar(i, body_size, width=0.6, bottom=body_bottom,
                   color=color, alpha=0.9, edgecolor=color, linewidth=0.5, zorder=3)
        
        # Pivots
        for p in result['pivots']['high']:
            idx = p['index'] - offset
            if 0 <= idx < len(df):
                ax.plot(idx, p['price'], marker='^', color='#8B0000', 
                       markersize=9, zorder=5)
                ax.text(idx, p['price'] + price_range * 0.025, p['type'],
                       fontsize=7, ha='center', color='#8B0000', fontweight='bold')
        
        for p in result['pivots']['low']:
            idx = p['index'] - offset
            if 0 <= idx < len(df):
                ax.plot(idx, p['price'], marker='v', color='#006400',
                       markersize=9, zorder=5)
                ax.text(idx, p['price'] - price_range * 0.025, p['type'],
                       fontsize=7, ha='center', color='#006400', fontweight='bold')
        
        # Structures
        for struct in result['structures']:
            idx = struct['index'] - offset
            if 0 <= idx < len(df):
                color = '#00CED1' if struct['type'] == 'CHoCH' else '#FFD700'
                ax.axvline(x=idx, color=color, linewidth=2, alpha=0.6, zorder=4)
        
        ax.set_xlabel(f'Bars (Last {display_bars})', fontsize=10)
        ax.set_ylabel('Price (USDT)', fontsize=10)
        ax.set_title(f'SMC Analysis - Swing Length: {swing_length}', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-1, len(df))
        ax.set_ylim(price_min - price_range * 0.1, price_max + price_range * 0.1)
        ax.set_facecolor('#f8f9fa')
        
        self.fig.tight_layout()
        self.chart_canvas.draw()


def main():
    root = tk.Tk()
    app = ModelEnsembleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
