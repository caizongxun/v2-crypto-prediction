"""
GUI 訓練介面
點擊按鈕影葲模型訓練
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from data import load_btc_data
from indicators import IndicatorCalculator
from feature_engineering_v2 import AdvancedFeatureEngineering
from train_models_v2 import ModelTrainerV2
from config import HF_TOKEN


class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("加密貨幣預測模型訓練器")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        self.training_thread = None
        self.is_training = False
        
        self.setup_ui()
    
    def setup_ui(self):
        # 主控制區
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 樣次
        ttk.Label(control_frame, text="選擇訓練模式:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.mode_var = tk.StringVar(value="v2")
        modes = [
            ("V2 日常訓練", "v2"),
            ("V2.1 指標修複", "v2.1"),
            ("V2.2 特徵优化", "v2.2"),
        ]
        
        for text, value in modes:
            ttk.Radiobutton(control_frame, text=text, variable=self.mode_var, value=value).grid(
                row=0, column=modes.index((text, value))+1, padx=10
            )
        
        # 分隔符
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 詳變檻
        info_frame = ttk.LabelFrame(self.root, text="訓練信息", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        # 狀態行
        status_row = ttk.Frame(info_frame)
        status_row.pack(fill=tk.X, pady=5)
        ttk.Label(status_row, text="訓練狀態:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_row, text="将待鼀活", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # 進度条
        progress_row = ttk.Frame(info_frame)
        progress_row.pack(fill=tk.X, pady=5)
        ttk.Label(progress_row, text="進度:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_row, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.progress_label = ttk.Label(progress_row, text="0%")
        self.progress_label.pack(side=tk.LEFT)
        
        # 準確率行
        accuracy_row = ttk.Frame(info_frame)
        accuracy_row.pack(fill=tk.X, pady=5)
        ttk.Label(accuracy_row, text="方向準確率:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.accuracy_label = ttk.Label(accuracy_row, text="-", font=("Arial", 10, "bold"), foreground="blue")
        self.accuracy_label.pack(side=tk.LEFT, padx=10)
        
        # 按鈕流
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.train_button = ttk.Button(
            button_frame, text="開始訓練", command=self.start_training,
            width=20, state=tk.NORMAL
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(
            button_frame, text="停止", command=self.cancel_training,
            width=20, state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, text="檢查結果", command=self.show_results,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, text="清除日誌", command=self.clear_logs,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # 詳變檻
        log_frame = ttk.LabelFrame(self.root, text="訓練日誌", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=15, width=100, bg="white", font=("Courier", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self.log("[INFO] 系統准備完成")
        self.log("[INFO] 選擇訓練模式並點擊[\u958b始訓練]按鈕")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_progress(self, value):
        self.progress_var.set(value)
        self.progress_label.config(text=f"{int(value)}%")
        self.root.update()
    
    def update_accuracy(self, accuracy):
        self.accuracy_label.config(text=f"{accuracy:.2%}")
        if accuracy >= 0.70:
            self.accuracy_label.config(foreground="green")
        elif accuracy >= 0.60:
            self.accuracy_label.config(foreground="orange")
        else:
            self.accuracy_label.config(foreground="red")
        self.root.update()
    
    def start_training(self):
        mode = self.mode_var.get()
        self.train_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.status_label.config(text="訓練中...", foreground="blue")
        self.progress_var.set(0)
        
        self.training_thread = threading.Thread(
            target=self.train_worker, args=(mode,), daemon=True
        )
        self.training_thread.start()
    
    def train_worker(self, mode):
        try:
            self.log(f"[START] 開始 {mode} 訓練...")
            self.log(f"[INFO] 訓練模式: {mode}")
            
            # 1. 載入數據
            self.log("[1/5] 正在載入數據...")
            self.update_progress(10)
            df = load_btc_data(hf_token=HF_TOKEN)
            if df is None:
                self.log("[ERROR] 載入數據失敗")
                return
            self.log(f"[OK] 數據載入完成: {df.shape[0]} 根 K 線")
            
            # 2. 計算技術指標
            self.log("[2/5] 正在計算技術指標...")
            self.update_progress(30)
            calc = IndicatorCalculator()
            indicators = calc.calculate_all(df)
            self.log(f"[OK] 技術指標計算完成: {len(indicators)} 個")
            
            # 3. 構建特徵
            self.log("[3/5] 正在構建特徵...")
            self.update_progress(50)
            
            trend_strength = np.ones(len(df)) * 0.5
            volatility_index = np.ones(len(df)) * 0.5
            direction_confirmation = np.ones(len(df)) * 0.5
            
            formulas_outputs = {
                'trend_strength': trend_strength,
                'volatility_index': volatility_index,
                'direction_confirmation': direction_confirmation
            }
            
            trainer = ModelTrainerV2(df, {})
            X, y = trainer.prepare_features_v2(indicators, formulas_outputs)
            self.log(f"[OK] 特徵構建完成: {X.shape[1]} 個特徵")
            
            # 4. 訓練模型
            self.log("[4/5] 正在訓練模型...")
            self.update_progress(70)
            results = trainer.train_v2(X, y)
            
            accuracy = results['direction_accuracy']
            self.log(f"[OK] 模型訓練完成")
            self.log(f"[OK] 方向準確率: {accuracy:.2%}")
            self.log(f"[OK] 相比 v1 提升: +{results['improvement_from_v1']:.2f}%")
            
            # 5. 保存模型
            self.log("[5/5] 正在保存模型...")
            self.update_progress(90)
            trainer.save_models_v2()
            self.log("[OK] 模型保存完成")
            
            # 完成
            self.update_progress(100)
            self.update_accuracy(accuracy)
            self.log("[SUCCESS] 訓練完成!")
            self.status_label.config(text="完成", foreground="green")
            
            # 淘氣且匪保的优化提議
            if accuracy < 0.60:
                self.log("[TIP] 准確率仍有提升空間, 建議定次檢查模型配置")
            elif accuracy < 0.70:
                self.log("[TIP] 空間接近目標, 可以繼續优化")
            else:
                self.log("[SUCCESS] 已達到目標!")
            
        except Exception as e:
            self.log(f"[ERROR] {str(e)}")
            self.status_label.config(text="错誤", foreground="red")
        finally:
            self.train_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
    
    def cancel_training(self):
        messagebox.showinfo("提示", "程序將漸進弊停")
        self.status_label.config(text="已停止", foreground="orange")
    
    def show_results(self):
        results_file = Path("models_v2/training_results_v2.json")
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            result_window = tk.Toplevel(self.root)
            result_window.title("訓練結果")
            result_window.geometry("500x400")
            
            text = scrolledtext.ScrolledText(result_window, height=20, width=60, bg="white")
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            content = "\n".join([
                f"{key}: {value}"
                for key, value in results.items()
            ])
            text.insert(tk.END, json.dumps(results, indent=2, ensure_ascii=False))
            text.config(state=tk.DISABLED)
        else:
            messagebox.showwarning("提示", "沒有找到訓練結果")
    
    def clear_logs(self):
        if messagebox.askyesno("確認", "突撕清除日誌?"):
            self.log_text.delete(1.0, tk.END)
            self.log("[INFO] 日誌已清除")


def main():
    root = tk.Tk()
    gui = TrainingGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
