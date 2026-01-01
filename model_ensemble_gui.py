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
import requests
import json
from typing import Dict, Optional
import threading
import os

# Set matplotlib font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 100

# API Configuration
# 注意: 請確保 API Key 完整且有效
# 可在 https://console.groq.com/keys 生成新的 Key
GROQ_API_KEY = 'gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'  # 請替換為你的 API Key


class PineScriptAIConverter:
    """Groq AI powered PineScript to Python converter"""
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = GROQ_API_KEY
        self.api_key = api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        # 使用最新的可用模型列表 (2026年1月)
        # llama-3.1-70b-versatile 已下架，改用 llama-3.3-70b-versatile
        self.models = [
            "llama-3.3-70b-versatile",      # 最新推薦模型
            "llama-3.1-8b-instant",         # 快速輕量版本
            "mixtral-8x7b-32768",           # 穩定版本
            "gemma2-9b-it",                 # 備選模型
        ]
        self.current_model = self.models[0]
    
    def _prepare_prompt(self, pinescript_code: str) -> str:
        """準備簡化版本的轉換提示詞"""
        prompt = f"""Convert this PineScript v5 code to Python using pandas and numpy.

Provide output as JSON with keys:
- python_code: the converted Python code
- explanation: brief explanation of what the indicator does
- warnings: any uncertain conversions

PineScript code:
{pinescript_code}

Respond only with valid JSON."""
        return prompt
    
    def convert(self, pinescript_code: str) -> Dict:
        """Use Groq API to convert PineScript to Python"""
        if not self.api_key or self.api_key == 'gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX':
            return {
                "error": "API Key 未配置",
                "warning": "請設置有效的 GROQ_API_KEY",
                "help": "訪問 https://console.groq.com/keys 獲取 API Key"
            }
        
        # 嘗試多個模型
        for model in self.models:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": self._prepare_prompt(pinescript_code)
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048
                }
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                # 成功響應
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # 嘗試解析 JSON
                    try:
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            parsed = json.loads(content[json_start:json_end])
                            parsed['model_used'] = model
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    
                    return {
                        "raw_response": content,
                        "note": "無法解析為 JSON，返回原始響應",
                        "model_used": model
                    }
                
                # 401 Unauthorized - API Key 問題
                elif response.status_code == 401:
                    return {
                        "error": "認證失敗 (401)",
                        "details": "API Key 無效或已過期",
                        "help": "請檢查 API Key 是否正確",
                        "attempted_model": model
                    }
                
                # 429 Too Many Requests - 嘗試下一個模型
                elif response.status_code == 429:
                    continue
                
                # 400 Bad Request - 模型不可用，嘗試下一個
                elif response.status_code == 400:
                    error_text = response.text
                    # 檢查是否是模型下架的錯誤
                    if "decommissioned" in error_text.lower():
                        continue  # 嘗試下一個模型
                    else:
                        continue  # 也嘗試下一個
                
                else:
                    continue  # 嘗試下一個模型
                    
            except requests.exceptions.Timeout:
                continue  # 嘗試下一個模型
            except requests.exceptions.ConnectionError:
                return {
                    "error": "網路連接失敗",
                    "details": "無法連接到 Groq API",
                    "help": "請檢查網路連接和防火牆設置"
                }
            except Exception as e:
                continue  # 嘗試下一個模型
        
        # 所有模型都失敗
        return {
            "error": "所有模型請求都失敗",
            "details": "嘗試了多個模型但都無法成功",
            "help": "請檢查 API Key 有效性和 Groq 服務狀態",
            "attempted_models": self.models
        }


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
        self.root.title('加密貨幣預測系統 + 聰明錢概念')
        self.root.geometry('1400x850')
        
        self.df = None
        self.models = {}
        self.converter = None
        self.last_conversion_result = None
        
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
        
        self.converter_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.converter_frame, text='PineScript 轉換')
        self.setup_converter_tab()
        
        # Initialize converter with default API key
        self.converter = PineScriptAIConverter(GROQ_API_KEY)

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

    def setup_converter_tab(self):
        """PineScript to Python Converter Tab"""
        main_frame = ttk.Frame(self.converter_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # API Status
        status_frame = ttk.LabelFrame(main_frame, text='API 狀態', padding=10)
        status_frame.pack(fill=tk.X, pady=10)
        
        api_status = '未設置' if GROQ_API_KEY == 'gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' else '已配置'
        status_color = 'red' if api_status == '未設置' else 'green'
        
        ttk.Label(status_frame, text=f'Groq API: {api_status}', 
                 foreground=status_color, font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(status_frame, text='(llama-3.3-70b-versatile)', 
                 foreground='gray', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(status_frame, text='測試連接', 
                  command=self.test_groq_connection).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(status_frame, text='API Key 設置幫助', 
                  command=self.show_api_help).pack(side=tk.LEFT, padx=5)
        
        # Input/Output Frame
        io_frame = ttk.Frame(main_frame)
        io_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Input Section
        input_frame = ttk.LabelFrame(io_frame, text='PineScript 代碼輸入', padding=10)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.input_text = tk.Text(input_frame, height=25, width=50, wrap=tk.WORD)
        scrollbar_input = ttk.Scrollbar(input_frame, orient=tk.VERTICAL, command=self.input_text.yview)
        self.input_text.config(yscrollcommand=scrollbar_input.set)
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_input.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text='轉換', 
                  command=self.convert_pinescript).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='從文件加載', 
                  command=self.load_pinescript_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='保存結果', 
                  command=self.save_conversion_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='清空', 
                  command=lambda: self.input_text.delete('1.0', tk.END)).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(button_frame, text='就緒', foreground='green')
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Output Section
        output_frame = ttk.LabelFrame(io_frame, text='轉換結果', padding=10)
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.output_text = tk.Text(output_frame, height=25, width=50, wrap=tk.WORD)
        scrollbar_output = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=scrollbar_output.set)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_output.pack(side=tk.RIGHT, fill=tk.Y)

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

    def load_local_data(self):
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("CSV 文件", "*.csv"), ("Parquet 文件", "*.parquet")]
            )
            if filepath:
                if filepath.endswith('.csv'):
                    self.df = pd.read_csv(filepath)
                else:
                    self.df = pd.read_parquet(filepath)
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
            self.update_load_status()
            messagebox.showinfo('成功', f'已加載 {len(self.df)} 行')
        except Exception as e:
            messagebox.showerror('錯誤', f'失敗: {str(e)}')

    def update_load_status(self):
        if self.df is not None:
            self.load_status.config(text=f'已加載: {len(self.df)} 行', foreground='green')
            info = f"行數: {len(self.df)}\n列: {', '.join(self.df.columns[:5])}"
            self.load_info.config(text=info)

    def analyze_smc(self):
        if self.df is None:
            messagebox.showwarning('警告', '請先加載數據')
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
            
            msg = (f"高樞紐點: {high_pivots}\n"
                   f"低樞紐點: {low_pivots}\n"
                   f"結構: {structures}\n"
                   f"訂單區塊: {obs}")
            messagebox.showinfo('完成', msg)
        except Exception as e:
            messagebox.showerror('錯誤', f'{str(e)}')
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
        
        ax.set_xlabel(f'柱線 (最後 {display_bars} 根)', fontsize=10)
        ax.set_ylabel('價格 (USDT)', fontsize=10)
        ax.set_title(f'SMC 分析 - 擺幅長度: {swing_length}', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-1, len(df))
        ax.set_ylim(price_min - price_range * 0.1, price_max + price_range * 0.1)
        ax.set_facecolor('#f8f9fa')
        
        self.fig.tight_layout()
        self.chart_canvas.draw()

    # PineScript Converter Methods
    def show_api_help(self):
        """顯示 API Key 設置幫助"""
        help_text = """如何設置 Groq API Key:

1. 訪問 https://console.groq.com/keys

2. 如果沒有帳號，點擊註冊

3. 登錄後，點擊 "Create New API Key"

4. 複製生成的 API Key

5. 編輯 model_ensemble_gui.py，找到這一行：
   GROQ_API_KEY = 'gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

6. 替換為你的實際 Key：
   GROQ_API_KEY = 'gsk_你的實際KEY'

7. 保存文件後重新運行程序

最新的模型: llama-3.3-70b-versatile

常見問題:
- 確保 API Key 完整無誤
- 檢查網路連接
- 嘗試測試連接按鈕驗證 Key 有效性
        """
        messagebox.showinfo('API Key 設置指南', help_text)
    
    def test_groq_connection(self):
        """Test Groq API connection"""
        if not self.converter:
            self.converter = PineScriptAIConverter(GROQ_API_KEY)
        
        self.status_label.config(text='測試連接中...', foreground='blue')
        self.root.update()
        
        try:
            test_code = "length = input(14, 'Period')"
            result = self.converter.convert(test_code)
            
            if 'error' in result:
                self.status_label.config(text='連接失敗', foreground='red')
                error_msg = f"{result.get('error', '')}\n\n{result.get('details', '')}\n\n{result.get('help', '')}"
                messagebox.showerror('錯誤', error_msg)
            else:
                self.status_label.config(text='連接成功', foreground='green')
                model_used = result.get('model_used', 'unknown')
                messagebox.showinfo('成功', f'Groq API 連接成功\n\n使用模型: {model_used}')
        except Exception as e:
            self.status_label.config(text='錯誤', foreground='red')
            messagebox.showerror('錯誤', f'連接檢查失敗: {str(e)}')
    
    def load_pinescript_file(self):
        """Load PineScript code from file"""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("PineScript 文件", "*.pine"), ("文本文件", "*.txt"), ("所有文件", "*.*")]
            )
            if filepath:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()
                self.input_text.delete('1.0', tk.END)
                self.input_text.insert('1.0', code)
                self.status_label.config(text='文件已加載', foreground='green')
        except Exception as e:
            messagebox.showerror('錯誤', f'加載文件失敗: {str(e)}')
    
    def convert_pinescript(self):
        """Convert PineScript to Python"""
        if not self.converter:
            messagebox.showwarning('警告', '轉換器未初始化')
            return
        
        code = self.input_text.get('1.0', tk.END).strip()
        if not code:
            messagebox.showwarning('警告', '請輸入 PineScript 代碼')
            return
        
        self.status_label.config(text='轉換中...', foreground='blue')
        self.root.update()
        
        # Run conversion in background thread
        thread = threading.Thread(target=self._do_conversion, args=(code,))
        thread.daemon = True
        thread.start()
    
    def _do_conversion(self, code):
        """Background conversion task"""
        try:
            result = self.converter.convert(code)
            
            # Display result
            self.output_text.delete('1.0', tk.END)
            
            # Format output
            output = json.dumps(result, indent=2, ensure_ascii=False)
            self.output_text.insert('1.0', output)
            
            # Store result for saving
            self.last_conversion_result = result
            
            self.status_label.config(text='轉換完成', foreground='green')
            if 'error' not in result:
                messagebox.showinfo('成功', '轉換成功完成')
        except Exception as e:
            self.status_label.config(text='轉換錯誤', foreground='red')
            messagebox.showerror('錯誤', f'轉換失敗: {str(e)}')
    
    def save_conversion_result(self):
        """Save conversion result to file"""
        output = self.output_text.get('1.0', tk.END).strip()
        if not output:
            messagebox.showwarning('警告', '無結果可保存')
            return
        
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON 文件", "*.json"), ("Python 文件", "*.py"), ("文本文件", "*.txt")]
            )
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(output)
                messagebox.showinfo('成功', f'結果已保存到 {filepath}')
        except Exception as e:
            messagebox.showerror('錯誤', f'保存失敗: {str(e)}')


def main():
    root = tk.Tk()
    app = ModelEnsembleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
