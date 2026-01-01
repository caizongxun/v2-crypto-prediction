import re
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
from datetime import datetime


class PineScriptConverter:
    """
    Pine Script 到 Python 的轉換器
    支援基本的 Pine Script 語法轉換
    """
    
    def __init__(self):
        self.variable_map = {}  # 追蹤變數映射
        self.functions_map = {}  # 內置函數映射
        self.plot_commands = []  # 繪圖命令
        self.setup_mappings()
    
    def setup_mappings(self):
        """設置 Pine Script 到 Python 的映射"""
        self.functions_map = {
            'close': 'df["close"]',
            'open': 'df["open"]',
            'high': 'df["high"]',
            'low': 'df["low"]',
            'volume': 'df["volume"]',
            'ta.sma': 'self.sma',
            'ta.ema': 'self.ema',
            'ta.rsi': 'self.rsi',
            'ta.macd': 'self.macd',
            'ta.bollinger': 'self.bollinger_bands',
            'ta.atr': 'self.atr',
            'ta.stoch': 'self.stochastic',
            'math.sum': 'np.sum',
            'math.avg': 'np.mean',
            'math.max': 'np.max',
            'math.min': 'np.min',
            'math.abs': 'np.abs',
            'math.sqrt': 'np.sqrt',
            'math.pow': 'np.power',
            'math.log': 'np.log',
            'str.tostring': 'str',
            'nz': 'self.nz',  # 處理 NaN/None
        }
    
    def convert_pine_to_python(self, pine_code):
        """
        將 Pine Script 代碼轉換為 Python
        """
        lines = pine_code.split('\n')
        python_code = []
        
        # 添加必要的導入
        imports = [
            'import numpy as np',
            'import pandas as pd',
            'from typing import List, Dict, Tuple',
        ]
        python_code.extend(imports)
        python_code.append('')
        
        # 解析 indicator() 聲明
        indicator_info = self.extract_indicator_info(pine_code)
        python_code.append(f'class CustomIndicator:')
        python_code.append(f'    """')
        python_code.append(f'    Indicator: {indicator_info.get("title", "Custom Indicator")}')
        python_code.append(f'    Description: {indicator_info.get("description", "")}')
        python_code.append(f'    """')
        python_code.append('')
        
        python_code.append('    def __init__(self):')
        python_code.append('        self.plots = {}')
        python_code.append('        self.signals = {}')
        python_code.append('')
        
        # 轉換 input() 定義
        input_lines = self.extract_inputs(pine_code)
        if input_lines:
            python_code.append('    def setup_parameters(self):')
            for input_line in input_lines:
                python_code.append(f'        {input_line}')
            python_code.append('')
        
        # 轉換主邏輯
        python_code.append('    def calculate(self, df: pd.DataFrame) -> Dict:')
        python_code.append('        """計算指標值"""')
        python_code.append('        results = {')
        python_code.append('            "signals": [],')
        python_code.append('            "plots": {},')
        python_code.append('        }')
        python_code.append('')
        
        # 轉換變數聲明和邏輯
        converted_lines = self.convert_logic(lines)
        for line in converted_lines:
            if line.strip():
                python_code.append(f'        {line}')
        
        python_code.append('        return results')
        python_code.append('')
        
        # 添加輔助函數
        python_code.extend(self.generate_helper_functions())
        
        return '\n'.join(python_code)
    
    def extract_indicator_info(self, code):
        """提取 indicator() 信息"""
        info = {}
        
        # 提取 title
        title_match = re.search(r'title\s*=\s*["\']([^"\']*)["\']
', code)
        if title_match:
            info['title'] = title_match.group(1)
        
        # 提取 description
        desc_match = re.search(r'overlay\s*=\s*(true|false)', code, re.IGNORECASE)
        if desc_match:
            info['overlay'] = desc_match.group(1).lower() == 'true'
        
        return info
    
    def extract_inputs(self, code):
        """提取 input() 參數定義"""
        inputs = []
        input_pattern = r'(\w+)\s*=\s*input\.?(\w*)\s*\(([^)]*)\)'
        
        for match in re.finditer(input_pattern, code):
            var_name = match.group(1)
            input_type = match.group(2)
            params = match.group(3)
            
            # 解析參數
            default_val = self.extract_default_value(params)
            
            if input_type.lower() == 'int':
                inputs.append(f'self.{var_name} = {default_val}  # integer input')
            elif input_type.lower() == 'float':
                inputs.append(f'self.{var_name} = {default_val}  # float input')
            elif input_type.lower() == 'string':
                inputs.append(f'self.{var_name} = "{default_val}"  # string input')
            elif input_type.lower() == 'bool':
                inputs.append(f'self.{var_name} = {str(default_val).lower()}  # boolean input')
            else:
                inputs.append(f'self.{var_name} = {default_val}')
        
        return inputs
    
    def extract_default_value(self, params):
        """提取默認值"""
        match = re.search(r'defval\s*=\s*([^,\)]+)', params)
        if match:
            val = match.group(1).strip()
            return val
        return '0'
    
    def convert_logic(self, lines):
        """轉換主邏輯部分"""
        converted = []
        in_function = False
        function_indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            # 跳過註解和空行
            if not stripped or stripped.startswith('//'):
                continue
            
            # 跳過 Pine Script 特定的聲明
            if any(x in stripped for x in ['@version', 'indicator(', 'input(', 'plot(']):
                if 'plot(' in stripped:
                    # 記錄 plot 命令
                    self.parse_plot_command(stripped)
                continue
            
            # 轉換變數聲明
            if 'var ' in stripped:
                converted_line = stripped.replace('var ', '')
                converted.append(converted_line)
            
            # 轉換 if/else
            elif stripped.startswith('if '):
                condition = self.convert_condition(stripped[3:].rstrip(':'))
                converted.append(f'if {condition}:')
            
            elif stripped.startswith('else'):
                converted.append('else:')
            
            # 轉換函數調用
            elif any(func in stripped for func in self.functions_map.keys()):
                converted_line = self.convert_functions(stripped)
                converted.append(converted_line)
            
            # 其他語句
            else:
                converted.append(stripped)
        
        return converted
    
    def convert_condition(self, condition):
        """轉換條件語句"""
        # 替換 Pine Script 邏輯運算符
        converted = condition
        converted = converted.replace('and', 'and')
        converted = converted.replace('or', 'or')
        converted = converted.replace('not', 'not')
        
        # 替換函數調用
        for pine_func, py_func in self.functions_map.items():
            converted = converted.replace(pine_func, py_func)
        
        return converted
    
    def convert_functions(self, statement):
        """轉換函數調用"""
        result = statement
        
        for pine_func, py_func in self.functions_map.items():
            if pine_func in result:
                # 簡單替換,避免過度復雜的正則
                result = result.replace(f'{pine_func}(', f'{py_func}(')
        
        return result
    
    def parse_plot_command(self, plot_line):
        """解析 plot() 命令"""
        # 提取繪圖信息
        match = re.search(r'plot\(([^,]+),\s*([^,]+),\s*color=([^,)]+)', plot_line)
        if match:
            series = match.group(1).strip()
            label = match.group(2).strip().strip('"')
            color = match.group(3).strip()
            
            self.plot_commands.append({
                'series': series,
                'label': label,
                'color': color,
            })
    
    def generate_helper_functions(self):
        """生成輔助函數"""
        helpers = [
            '',
            '    def nz(self, value, replacement=0):',
            '        """處理 NaN 值"""',
            '        if pd.isna(value):',
            '            return replacement',
            '        return value',
            '',
            '    def sma(self, data, length):',
            '        """簡單移動平均"""',
            '        return data.rolling(window=length).mean()',
            '',
            '    def ema(self, data, length):',
            '        """指數移動平均"""',
            '        return data.ewm(span=length, adjust=False).mean()',
            '',
            '    def rsi(self, data, length=14):',
            '        """相對強度指數"""',
            '        delta = data.diff()',
            '        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()',
            '        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()',
            '        rs = gain / loss',
            '        rsi = 100 - (100 / (1 + rs))',
            '        return rsi',
            '',
            '    def macd(self, data, fast=12, slow=26, signal=9):',
            '        """MACD 指標"""',
            '        ema_fast = data.ewm(span=fast, adjust=False).mean()',
            '        ema_slow = data.ewm(span=slow, adjust=False).mean()',
            '        macd_line = ema_fast - ema_slow',
            '        signal_line = macd_line.ewm(span=signal, adjust=False).mean()',
            '        histogram = macd_line - signal_line',
            '        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}',
            '',
            '    def bollinger_bands(self, data, length=20, deviation=2):',
            '        """布林帶"""',
            '        sma = data.rolling(window=length).mean()',
            '        std = data.rolling(window=length).std()',
            '        upper = sma + (std * deviation)',
            '        lower = sma - (std * deviation)',
            '        return {"upper": upper, "middle": sma, "lower": lower}',
            '',
            '    def atr(self, high, low, close, length=14):',
            '        """平均真實波幅"""',
            '        tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))',
            '        atr = pd.Series(tr).rolling(window=length).mean()',
            '        return atr',
            '',
            '    def stochastic(self, high, low, close, k_period=14, d_period=3):',
            '        """隨機指標"""',
            '        lowest_low = low.rolling(window=k_period).min()',
            '        highest_high = high.rolling(window=k_period).max()',
            '        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))',
            '        d_percent = k_percent.rolling(window=d_period).mean()',
            '        return {"k": k_percent, "d": d_percent}',
        ]
        
        return helpers


class IndicatorConverter:
    """通用指標轉換工具"""
    
    def __init__(self, root):
        self.root = root
        self.root.title('Indicator Source Code Converter')
        self.root.geometry('1200x800')
        
        self.pine_converter = PineScriptConverter()
        self.setup_ui()
    
    def setup_ui(self):
        """設置用戶界面"""
        # 頂部控制區
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text='Indicator Type:').pack(side=tk.LEFT, padx=5)
        
        self.indicator_type = ttk.Combobox(
            control_frame,
            values=['Pine Script', 'Custom Python'],
            state='readonly',
            width=20
        )
        self.indicator_type.set('Pine Script')
        self.indicator_type.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text='Convert',
                  command=self.convert_code).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text='Clear All',
                  command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # 主要區域 - 分成左右兩個面板
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側 - 輸入區
        left_frame = ttk.LabelFrame(main_frame, text='Source Code (Input)', padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, height=30)
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        # 右側 - 輸出區
        right_frame = ttk.LabelFrame(main_frame, text='Converted Python Code (Output)', padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.output_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=30, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # 底部狀態欄
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(status_frame, text='Copy Output',
                  command=self.copy_output).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(status_frame, text='Save Python File',
                  command=self.save_python_file).pack(side=tk.RIGHT, padx=5)
        
        self.status_label = ttk.Label(status_frame, text='Ready', foreground='green')
        self.status_label.pack(side=tk.LEFT, padx=5)
    
    def convert_code(self):
        """轉換代碼"""
        input_code = self.input_text.get(1.0, tk.END)
        
        if not input_code.strip():
            messagebox.showwarning('Warning', 'Please enter source code first')
            return
        
        try:
            indicator_type = self.indicator_type.get()
            
            if indicator_type == 'Pine Script':
                output_code = self.pine_converter.convert_pine_to_python(input_code)
            else:
                output_code = input_code  # TODO: 其他轉換類型
            
            # 更新輸出
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(1.0, output_code)
            self.output_text.config(state=tk.DISABLED)
            
            self.status_label.config(text='Conversion successful', foreground='green')
        
        except Exception as e:
            messagebox.showerror('Error', f'Conversion failed: {str(e)}')
            self.status_label.config(text=f'Error: {str(e)}', foreground='red')
    
    def clear_all(self):
        """清空所有內容"""
        self.input_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.status_label.config(text='Cleared', foreground='gray')
    
    def copy_output(self):
        """複製輸出"""
        try:
            output = self.output_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(output)
            self.status_label.config(text='Copied to clipboard', foreground='blue')
        except Exception as e:
            messagebox.showerror('Error', f'Copy failed: {str(e)}')
    
    def save_python_file(self):
        """保存為 Python 文件"""
        try:
            from tkinter import filedialog
            filepath = filedialog.asksaveasfilename(
                defaultextension='.py',
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            
            if filepath:
                output = self.output_text.get(1.0, tk.END)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(output)
                
                messagebox.showinfo('Success', f'Saved to {filepath}')
                self.status_label.config(text=f'Saved: {filepath}', foreground='green')
        
        except Exception as e:
            messagebox.showerror('Error', f'Save failed: {str(e)}')


if __name__ == '__main__':
    root = tk.Tk()
    app = IndicatorConverter(root)
    root.mainloop()
