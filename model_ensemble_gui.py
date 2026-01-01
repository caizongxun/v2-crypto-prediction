import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib
from pathlib import Path

from fib_ob_fusion import FibonacciBollingerBands, OrderBlockDetector, FibOBFusion

# Set matplotlib font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100


class SmartMoneyStructure:
    """Smart Money Concepts - Corrected Structure Detection"""
    def __init__(self, swing_length=50, internal_length=5):
        self.swing_length = swing_length
        self.internal_length = internal_length
        
        self.BULLISH_LEG = 1
        self.BEARISH_LEG = 0
        self.BULLISH = 1
        self.BEARISH = -1

    def get_leg(self, df, size=50):
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
        leg = self.get_leg(df, size)
        h = df['high'].values
        l = df['low'].values
        n = len(df)
        
        pivots = {'high': [], 'low': []}
        
        last_high_price = None
        last_low_price = None
        
        for i in range(size + 1, n):
            if leg[i] != leg[i - 1]:
                
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
                        pivot_type = 'LH'
                    
                    pivots['low'].append({
                        'type': pivot_type,
                        'price': lowest_price,
                        'index': lowest_idx,
                        'bar_index': i - 1
                    })
                    last_low_price = lowest_price
                
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
        self.root.title('加密貨幣預測系統 + 聰明錢概念 + Fibonacci OB融合')
        self.root.geometry('1400x850')
        
        self.df = None
        self.models = {}
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.load_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.load_frame, text='數據加載')
        self.setup_load_tab()
        
        self.feature_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.feature_frame, text='特徵工程')
        self.setup_feature_tab()
        
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text='模型訓練')
        self.setup_train_tab()
        
        self.eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.eval_frame, text='模型評估')
        self.setup_eval_tab()
        
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text='預測')
        self.setup_predict_tab()
        
        self.smc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.smc_frame, text='聰明錢概念')
        self.setup_smc_tab()
        
        self.fusion_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.fusion_frame, text='Fibonacci OB融合')
        self.setup_fusion_tab()

    def setup_load_tab(self):
        frame = ttk.LabelFrame(self.load_frame, text='數據加載', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Button(frame, text='加載本地 CSV/Parquet', 
                  command=self.load_local_data).pack(pady=10)
        
        ttk.Button(frame, text='加載默認數據 (data/btc_15m.parquet)', 
                  command=self.load_default_data).pack(pady=10)
        
        self.load_status = ttk.Label(frame, text='未加載數據', foreground='red')
        self.load_status.pack(pady=10)
        
        self.load_info = ttk.Label(frame, text='', justify=tk.LEFT)
        self.load_info.pack(pady=10)

    def setup_feature_tab(self):
        frame = ttk.LabelFrame(self.feature_frame, text='特徵工程', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(frame, text='特徵開發進行中...').pack(pady=10)

    def setup_train_tab(self):
        frame = ttk.LabelFrame(self.train_frame, text='模型訓練', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(frame, text='模型開發進行中...').pack(pady=10)

    def setup_eval_tab(self):
        frame = ttk.LabelFrame(self.eval_frame, text='模型評估', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(frame, text='評估開發進行中...').pack(pady=10)

    def setup_predict_tab(self):
        frame = ttk.LabelFrame(self.predict_frame, text='預測', padding=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(frame, text='預測開發進行中...').pack(pady=10)

    def setup_smc_tab(self):
        frame = ttk.Frame(self.smc_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        param_frame = ttk.LabelFrame(frame, text='SMC 檢測參數', padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text='擺幅長度:').pack(side=tk.LEFT, padx=5)
        self.swing_length_spinbox = ttk.Spinbox(param_frame, from_=10, to=200, width=10)
        self.swing_length_spinbox.set(50)
        self.swing_length_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(param_frame, text='分析 SMC', 
                  command=self.analyze_smc).pack(side=tk.LEFT, padx=5)
        
        legend_frame = ttk.LabelFrame(frame, text='圖例', padding=10)
        legend_frame.pack(fill=tk.X, pady=5)
        ttk.Label(legend_frame, text='樞紐點: HH | HL | LL | LH', 
                 foreground='gray').pack()
        ttk.Label(legend_frame, text='OB: 藍色=看跌(HH->LL) | 綠色=看漲(LL->HH)', 
                 foreground='gray').pack()
        ttk.Label(legend_frame, text='結構: 黃色=BOS | 青色=CHoCH', 
                 foreground='gray').pack()
        
        chart_frame = ttk.Frame(frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.fig = Figure(figsize=(13, 6), dpi=100)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.smc_result = None

    def setup_fusion_tab(self):
        frame = ttk.Frame(self.fusion_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        param_frame = ttk.LabelFrame(frame, text='Fibonacci OB 融合參數', padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text='FBB 週期:').pack(side=tk.LEFT, padx=5)
        self.fib_length_spinbox = ttk.Spinbox(param_frame, from_=50, to=500, width=10)
        self.fib_length_spinbox.set(200)
        self.fib_length_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(param_frame, text='FBB 倍數:').pack(side=tk.LEFT, padx=5)
        self.fib_mult_spinbox = ttk.Spinbox(param_frame, from_=0.5, to=10.0, width=10, increment=0.1)
        self.fib_mult_spinbox.set(3.0)
        self.fib_mult_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(param_frame, text='OB 連續根數:').pack(side=tk.LEFT, padx=5)
        self.ob_periods_spinbox = ttk.Spinbox(param_frame, from_=3, to=20, width=10)
        self.ob_periods_spinbox.set(5)
        self.ob_periods_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(param_frame, text='分析融合', 
                  command=self.analyze_fusion).pack(side=tk.LEFT, padx=5)
        
        legend_frame = ttk.LabelFrame(frame, text='融合指標說明', padding=10)
        legend_frame.pack(fill=tk.X, pady=5)
        ttk.Label(legend_frame, text='Fibonacci Bollinger Bands: 中心線(紫) | 白線(0.236-0.764) | 紅線(1.0上限) | 綠線(1.0下限)', 
                 foreground='gray', wraplength=700, justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Label(legend_frame, text='Order Block: 藍色框=看跌OB | 綠色框=看漲OB | 三角符號=OB標記點', 
                 foreground='gray', wraplength=700, justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Label(legend_frame, text='融合信號: 當K線接近FBB水平且有OB存在時，訊號強度增加', 
                 foreground='gray', wraplength=700, justify=tk.LEFT).pack(anchor=tk.W)
        
        chart_frame = ttk.Frame(frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.fusion_fig = Figure(figsize=(13, 6.5), dpi=100)
        self.fusion_canvas = FigureCanvasTkAgg(self.fusion_fig, master=chart_frame)
        self.fusion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fusion_result = None

    def load_local_data(self):
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[('CSV 文件', '*.csv'), ('Parquet 文件', '*.parquet')]
            )
            if filepath:
                if filepath.endswith('.csv'):
                    self.df = pd.read_csv(filepath)
                else:
                    self.df = pd.read_parquet(filepath)
                self.df.columns = [col.lower() for col in self.df.columns]
                self.update_load_status()
                messagebox.showinfo('成功', f'已加載 {len(self.df)} 行')
        except Exception as e:
            messagebox.showerror('錯誤', f'加載失敗: {str(e)}')

    def load_default_data(self):
        try:
            path = 'data/btc_15m.parquet'
            if not Path(path).exists():
                path = 'btc_15m.parquet'
            self.df = pd.read_parquet(path)
            self.df.columns = [col.lower() for col in self.df.columns]
            self.update_load_status()
            messagebox.showinfo('成功', f'已加載 {len(self.df)} 行')
        except Exception as e:
            messagebox.showerror('錯誤', f'失敗: {str(e)}')

    def update_load_status(self):
        if self.df is not None:
            self.load_status.config(text=f'已加載: {len(self.df)} 行', foreground='green')
            cols_info = ', '.join(list(self.df.columns[:5]))
            date_range = f"{self.df.index[0]} ~ {self.df.index[-1]}" if len(self.df.index) > 0 else 'N/A'
            info = f"行數: {len(self.df)}\n列: {cols_info}\n日期範圍: {date_range}"
            self.load_info.config(text=info)

    def analyze_smc(self):
        if self.df is None:
            messagebox.showwarning('警告', '請先加載數據')
            return
        
        try:
            swing_length = int(self.swing_length_spinbox.get())
            smc = SmartMoneyStructure(swing_length=swing_length)
            self.smc_result = smc.analyze(self.df)
            self.plot_smc_analysis(swing_length)
            
            high_pivots = len(self.smc_result['pivots']['high'])
            low_pivots = len(self.smc_result['pivots']['low'])
            structures = len(self.smc_result['structures'])
            obs = len(self.smc_result['order_blocks'])
            
            stats = (f"高樞紐點 (HH+HL): {high_pivots}\n"
                    f"低樞紐點 (LL+LH): {low_pivots}\n"
                    f"結構打破 (BOS+CHoCH): {structures}\n"
                    f"訂單區塊: {obs}")
            messagebox.showinfo('SMC 分析完成', stats)
        except Exception as e:
            messagebox.showerror('錯誤', f'分析失敗: {str(e)}')
            import traceback
            traceback.print_exc()

    def analyze_fusion(self):
        if self.df is None:
            messagebox.showwarning('警告', '請先加載數據')
            return
        
        try:
            fib_length = int(self.fib_length_spinbox.get())
            fib_mult = float(self.fib_mult_spinbox.get())
            ob_periods = int(self.ob_periods_spinbox.get())
            
            fusion = FibOBFusion(fib_length=fib_length, fib_mult=fib_mult, ob_periods=ob_periods)
            self.fusion_result = fusion.analyze(self.df)
            
            self.plot_fusion_analysis()
            
            num_obs = len(self.fusion_result['ob']['all'])
            messagebox.showinfo('融合分析完成', f'檢測到 {num_obs} 個訂單區塊')
        except Exception as e:
            messagebox.showerror('錯誤', f'分析失敗: {str(e)}')
            import traceback
            traceback.print_exc()

    def plot_smc_analysis(self, swing_length: int):
        n = len(self.df)
        display_bars = min(300, n)
        offset = n - display_bars
        display_df = self.df.iloc[-display_bars:].reset_index(drop=True)
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        price_min = display_df['low'].min()
        price_max = display_df['high'].max()
        price_range = price_max - price_min
        
        for ob in self.smc_result['order_blocks']:
            display_start = ob['start_idx'] - offset
            display_end = ob['end_idx'] - offset
            
            if display_end >= 0 and display_start < len(display_df):
                plot_start = max(0, display_start)
                plot_end = min(len(display_df) - 1, display_end)
                width = plot_end - plot_start + 1
                
                if ob['is_mitigated']:
                    color = '#FF6B9D' if ob['type'] == 'bearish' else '#FFB3D9'
                    alpha = 0.2
                else:
                    color = '#4169E1' if ob['type'] == 'bearish' else '#32CD32'
                    alpha = 0.15
                
                rect = Rectangle(
                    (plot_start - 0.5, ob['low']),
                    width,
                    ob['high'] - ob['low'],
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha,
                    zorder=2
                )
                ax.add_patch(rect)
        
        for i in range(len(display_df)):
            o, h, l, c = display_df.loc[i, ['open', 'high', 'low', 'close']]
            color = '#00AA00' if c >= o else '#CC0000'
            
            ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=3, alpha=0.8)
            
            body_size = abs(c - o) if abs(c - o) > 0 else price_range * 0.001
            body_bottom = min(o, c)
            ax.bar(i, body_size, width=0.6, bottom=body_bottom,
                   color=color, alpha=0.9, edgecolor=color, linewidth=0.5, zorder=3)
        
        for p in self.smc_result['pivots']['high']:
            idx = p['index'] - offset
            if 0 <= idx < len(display_df):
                ax.plot(idx, p['price'], marker='^', color='#8B0000',
                       markersize=8, zorder=5, markeredgewidth=1)
                ax.text(idx, p['price'] + price_range * 0.03, p['type'],
                       fontsize=7, ha='center', color='#8B0000', fontweight='bold')
        
        for p in self.smc_result['pivots']['low']:
            idx = p['index'] - offset
            if 0 <= idx < len(display_df):
                ax.plot(idx, p['price'], marker='v', color='#006400',
                       markersize=8, zorder=5, markeredgewidth=1)
                ax.text(idx, p['price'] - price_range * 0.03, p['type'],
                       fontsize=7, ha='center', color='#006400', fontweight='bold')
        
        for struct in self.smc_result['structures']:
            idx = struct['index'] - offset
            if 0 <= idx < len(display_df):
                if struct['type'] == 'CHoCH':
                    color = '#00CED1'
                    linestyle = '--'
                else:
                    color = '#FFD700'
                    linestyle = '-'
                
                ax.axvline(x=idx, color=color, linewidth=1.5, alpha=0.6, linestyle=linestyle, zorder=4)
        
        ax.set_xlabel(f'K線索引 (最後 {display_bars} 根)', fontsize=10)
        ax.set_ylabel('價格 (USDT)', fontsize=10)
        ax.set_title(f'SMC 分析 - 擺幅長度: {swing_length}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-1, len(display_df))
        ax.set_ylim(price_min - price_range * 0.1, price_max + price_range * 0.1)
        ax.set_facecolor('#f8f9fa')
        
        legend_elements = [
            mpatches.Patch(color='#4169E1', alpha=0.15, label='看跌 OB'),
            mpatches.Patch(color='#32CD32', alpha=0.15, label='看漲 OB'),
            mpatches.Patch(color='#FFD700', alpha=0.6, label='BOS'),
            mpatches.Patch(color='#00CED1', alpha=0.6, label='CHoCH'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        self.fig.tight_layout()
        self.chart_canvas.draw()

    def plot_fusion_analysis(self):
        n = len(self.df)
        display_bars = min(300, n)
        offset = n - display_bars
        display_df = self.df.iloc[-display_bars:].reset_index(drop=True)
        
        self.fusion_fig.clear()
        ax = self.fusion_fig.add_subplot(111)
        
        price_min = display_df['low'].min()
        price_max = display_df['high'].max()
        price_range = price_max - price_min
        
        fib_result = self.fusion_result['fib']
        ob_result = self.fusion_result['ob']
        
        # Plot Order Blocks
        for ob in ob_result['all']:
            idx = ob['index'] - offset
            if 0 <= idx < len(display_df):
                color = '#4169E1' if ob['type'] == 'bearish' else '#32CD32'
                width = 1
                height = ob['high'] - ob['low']
                
                rect = Rectangle(
                    (idx - 0.5, ob['low']),
                    width,
                    height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.25,
                    zorder=2
                )
                ax.add_patch(rect)
                
                triangle_marker = '^' if ob['type'] == 'bullish' else 'v'
                ax.plot(idx, ob['avg'], marker=triangle_marker, color=color,
                       markersize=8, zorder=5, markeredgewidth=1)
        
        # Plot Candlesticks
        for i in range(len(display_df)):
            o, h, l, c = display_df.loc[i, ['open', 'high', 'low', 'close']]
            color = '#00AA00' if c >= o else '#CC0000'
            
            ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=3, alpha=0.8)
            
            body_size = abs(c - o) if abs(c - o) > 0 else price_range * 0.001
            body_bottom = min(o, c)
            ax.bar(i, body_size, width=0.6, bottom=body_bottom,
                   color=color, alpha=0.9, edgecolor=color, linewidth=0.5, zorder=3)
        
        # Plot Fibonacci Bollinger Bands
        basis = fib_result['basis'][-display_bars:]
        upper_618 = fib_result['upper_bands'][0.618][-display_bars:]
        lower_618 = fib_result['lower_bands'][0.618][-display_bars:]
        upper_1 = fib_result['upper_bands'][1.0][-display_bars:]
        lower_1 = fib_result['lower_bands'][1.0][-display_bars:]
        
        # Basis line (purple)
        ax.plot(basis, color='#FF00FF', linewidth=2, label='Basis (VWMA)', zorder=4)
        
        # 0.618 bands (white)
        ax.plot(upper_618, color='#CCCCCC', linewidth=1, alpha=0.7, zorder=3)
        ax.plot(lower_618, color='#CCCCCC', linewidth=1, alpha=0.7, zorder=3)
        
        # 1.0 bands (red/green)
        ax.plot(upper_1, color='#FF0000', linewidth=2, alpha=0.7, label='Upper 1.0', zorder=3)
        ax.plot(lower_1, color='#00AA00', linewidth=2, alpha=0.7, label='Lower 1.0', zorder=3)
        
        # Fill area between 0.618 bands
        ax.fill_between(range(len(upper_618)), upper_618, lower_618,
                        color='#FFFFFF', alpha=0.05, zorder=1)
        
        ax.set_xlabel(f'K線索引 (最後 {display_bars} 根)', fontsize=10)
        ax.set_ylabel('價格 (USDT)', fontsize=10)
        ax.set_title('Fibonacci Bollinger Bands + Order Block 融合分析', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-1, len(display_df))
        ax.set_ylim(price_min - price_range * 0.1, price_max + price_range * 0.1)
        ax.set_facecolor('#f8f9fa')
        
        legend_elements = [
            mpatches.Patch(color='#4169E1', alpha=0.25, label='看跌 OB'),
            mpatches.Patch(color='#32CD32', alpha=0.25, label='看漲 OB'),
            plt.Line2D([0], [0], color='#FF00FF', linewidth=2, label='Basis'),
            plt.Line2D([0], [0], color='#FF0000', linewidth=2, label='Upper 1.0'),
            plt.Line2D([0], [0], color='#00AA00', linewidth=2, label='Lower 1.0'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        self.fusion_fig.tight_layout()
        self.fusion_canvas.draw()


def main():
    root = tk.Tk()
    app = ModelEnsembleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
