"""
進階訓練器 - 參數調整 + LSTM 支援 (修復版本)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

from data import load_btc_data
from indicators import IndicatorCalculator
from feature_engineering_v2 import AdvancedFeatureEngineering
from config import HF_TOKEN


class AdvancedTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("進階訓練器 - 參數調整")
        self.root.geometry("1100x850")
        self.root.configure(bg="#f0f0f0")
        
        self.training_thread = None
        self.is_training = False
        self.model_cache = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        # 主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左邊 - 參數調整
        left_frame = ttk.LabelFrame(main_container, text="參數設定", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 模型選擇
        ttk.Label(left_frame, text="模型類型:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        self.model_var = tk.StringVar(value="lightgbm")
        for model in ["LightGBM", "XGBoost", "LSTM"]:
            ttk.Radiobutton(
                left_frame, text=model, variable=self.model_var, 
                value=model.lower(),
                command=self.update_params_ui
            ).pack(anchor=tk.W, padx=20)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 共通參數
        ttk.Label(left_frame, text="共通參數", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        
        self.params = {}
        
        # 訓練次數/Epochs
        ttk.Label(left_frame, text="訓練次數 (Epochs):").pack(anchor=tk.W, padx=20)
        self.epochs_var = tk.IntVar(value=100)
        self.epochs_scale = ttk.Scale(
            left_frame, from_=10, to=500, orient=tk.HORIZONTAL, 
            variable=self.epochs_var, command=lambda x: self.update_label("epochs")
        )
        self.epochs_scale.pack(anchor=tk.W, padx=20, fill=tk.X)
        self.epochs_label = ttk.Label(left_frame, text="100")
        self.epochs_label.pack(anchor=tk.W, padx=20)
        
        # 批次大小
        ttk.Label(left_frame, text="批次大小 (Batch Size):").pack(anchor=tk.W, padx=20, pady=(10, 0))
        self.batch_var = tk.IntVar(value=32)
        self.batch_scale = ttk.Scale(
            left_frame, from_=8, to=256, orient=tk.HORIZONTAL,
            variable=self.batch_var, command=lambda x: self.update_label("batch")
        )
        self.batch_scale.pack(anchor=tk.W, padx=20, fill=tk.X)
        self.batch_label = ttk.Label(left_frame, text="32")
        self.batch_label.pack(anchor=tk.W, padx=20)
        
        # 學習率
        ttk.Label(left_frame, text="學習率 (Learning Rate):").pack(anchor=tk.W, padx=20, pady=(10, 0))
        self.lr_var = tk.DoubleVar(value=0.05)
        self.lr_scale = ttk.Scale(
            left_frame, from_=0.001, to=0.1, orient=tk.HORIZONTAL,
            variable=self.lr_var, command=lambda x: self.update_label("lr")
        )
        self.lr_scale.pack(anchor=tk.W, padx=20, fill=tk.X)
        self.lr_label = ttk.Label(left_frame, text="0.0500")
        self.lr_label.pack(anchor=tk.W, padx=20)
        
        # 早停
        ttk.Label(left_frame, text="早停回合 (Early Stopping):").pack(anchor=tk.W, padx=20, pady=(10, 0))
        self.early_var = tk.IntVar(value=50)
        self.early_scale = ttk.Scale(
            left_frame, from_=10, to=200, orient=tk.HORIZONTAL,
            variable=self.early_var, command=lambda x: self.update_label("early")
        )
        self.early_scale.pack(anchor=tk.W, padx=20, fill=tk.X)
        self.early_label = ttk.Label(left_frame, text="50")
        self.early_label.pack(anchor=tk.W, padx=20)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 模型特定參數
        ttk.Label(left_frame, text="模型特定參數", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        self.model_params_frame = ttk.Frame(left_frame)
        self.model_params_frame.pack(anchor=tk.W, padx=20, fill=tk.X)
        
        self.update_params_ui()
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 優化建議
        ttk.Label(left_frame, text="優化建議", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=5)
        suggestions = [
            "當前準確率: 55.54%",
            "目標: 70%+",
            "",
            "LightGBM 推薦:",
            "  Epochs: 200-300",
            "  Max Depth: 10-12",
            "  Num Leaves: 63-127",
            "",
            "XGBoost 推薦:",
            "  Epochs: 150-200",
            "  Max Depth: 8-10",
            "",
            "LSTM 推薦:",
            "  Units: 64-128",
            "  Lookback: 20-30",
        ]
        for text in suggestions:
            ttk.Label(left_frame, text=text, font=("Courier", 8)).pack(anchor=tk.W, padx=20)
        
        # 右邊 - 訓練狀態和日誌
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 狀態面板
        status_frame = ttk.LabelFrame(right_frame, text="訓練狀態", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        # 狀態
        status_row = ttk.Frame(status_frame)
        status_row.pack(fill=tk.X, pady=5)
        ttk.Label(status_row, text="狀態:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_row, text="就緒", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # 進度條
        progress_row = ttk.Frame(status_frame)
        progress_row.pack(fill=tk.X, pady=5)
        ttk.Label(progress_row, text="進度:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_row, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.progress_label = ttk.Label(progress_row, text="0%", width=5)
        self.progress_label.pack(side=tk.LEFT)
        
        # 準確率
        accuracy_row = ttk.Frame(status_frame)
        accuracy_row.pack(fill=tk.X, pady=5)
        ttk.Label(accuracy_row, text="準確率:").pack(side=tk.LEFT)
        self.accuracy_label = ttk.Label(
            accuracy_row, text="-", font=("Arial", 11, "bold"), foreground="blue"
        )
        self.accuracy_label.pack(side=tk.LEFT, padx=10)
        
        # 訓練時間
        time_row = ttk.Frame(status_frame)
        time_row.pack(fill=tk.X, pady=5)
        ttk.Label(time_row, text="訓練時間:").pack(side=tk.LEFT)
        self.time_label = ttk.Label(time_row, text="-")
        self.time_label.pack(side=tk.LEFT, padx=10)
        
        # 按鈕
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.train_button = ttk.Button(
            button_frame, text="開始訓練", command=self.start_training
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame, text="停止", command=self.stop_training, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, text="清除日誌", command=self.clear_logs
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, text="保存參數", command=self.save_params
        ).pack(side=tk.LEFT, padx=5)
        
        # 日誌
        log_frame = ttk.LabelFrame(right_frame, text="訓練日誌", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=20, width=60, bg="white", font=("Courier", 8)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self.log("系統準備完成。調整參數後點擊開始訓練")
    
    def update_params_ui(self):
        # 清空舊的特定參數
        for widget in self.model_params_frame.winfo_children():
            widget.destroy()
        
        model = self.model_var.get()
        
        if model == "lightgbm":
            # LightGBM 特定參數
            ttk.Label(self.model_params_frame, text="樹深度 (Max Depth):").pack(anchor=tk.W)
            self.depth_var = tk.IntVar(value=8)
            ttk.Scale(self.model_params_frame, from_=4, to=20, orient=tk.HORIZONTAL,
                     variable=self.depth_var).pack(anchor=tk.W, fill=tk.X)
            
            ttk.Label(self.model_params_frame, text="葉子數 (Num Leaves):").pack(anchor=tk.W, pady=(10, 0))
            self.leaves_var = tk.IntVar(value=31)
            ttk.Scale(self.model_params_frame, from_=16, to=128, orient=tk.HORIZONTAL,
                     variable=self.leaves_var).pack(anchor=tk.W, fill=tk.X)
            
            ttk.Label(self.model_params_frame, text="正則化 (Reg Alpha):").pack(anchor=tk.W, pady=(10, 0))
            self.alpha_var = tk.DoubleVar(value=0.1)
            ttk.Scale(self.model_params_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                     variable=self.alpha_var).pack(anchor=tk.W, fill=tk.X)
        
        elif model == "xgboost":
            # XGBoost 特定參數
            ttk.Label(self.model_params_frame, text="樹深度 (Max Depth):").pack(anchor=tk.W)
            self.depth_var = tk.IntVar(value=6)
            ttk.Scale(self.model_params_frame, from_=3, to=15, orient=tk.HORIZONTAL,
                     variable=self.depth_var).pack(anchor=tk.W, fill=tk.X)
            
            ttk.Label(self.model_params_frame, text="採樣比例 (Subsample):").pack(anchor=tk.W, pady=(10, 0))
            self.subsample_var = tk.DoubleVar(value=0.8)
            ttk.Scale(self.model_params_frame, from_=0.5, to=1.0, orient=tk.HORIZONTAL,
                     variable=self.subsample_var).pack(anchor=tk.W, fill=tk.X)
        
        elif model == "lstm":
            # LSTM 特定參數
            ttk.Label(self.model_params_frame, text="隱藏層大小 (Units):").pack(anchor=tk.W)
            self.units_var = tk.IntVar(value=64)
            ttk.Scale(self.model_params_frame, from_=16, to=256, orient=tk.HORIZONTAL,
                     variable=self.units_var).pack(anchor=tk.W, fill=tk.X)
            
            ttk.Label(self.model_params_frame, text="序列長度 (Lookback):").pack(anchor=tk.W, pady=(10, 0))
            self.lookback_var = tk.IntVar(value=20)
            ttk.Scale(self.model_params_frame, from_=5, to=50, orient=tk.HORIZONTAL,
                     variable=self.lookback_var).pack(anchor=tk.W, fill=tk.X)
            
            ttk.Label(self.model_params_frame, text="丟棄率 (Dropout):").pack(anchor=tk.W, pady=(10, 0))
            self.dropout_var = tk.DoubleVar(value=0.2)
            ttk.Scale(self.model_params_frame, from_=0, to=0.5, orient=tk.HORIZONTAL,
                     variable=self.dropout_var).pack(anchor=tk.W, fill=tk.X)
    
    def update_label(self, param_type):
        if param_type == "epochs":
            self.epochs_label.config(text=str(self.epochs_var.get()))
        elif param_type == "batch":
            self.batch_label.config(text=str(self.batch_var.get()))
        elif param_type == "lr":
            self.lr_label.config(text=f"{self.lr_var.get():.4f}")
        elif param_type == "early":
            self.early_label.config(text=str(self.early_var.get()))
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_training(self):
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="訓練中...", foreground="blue")
        self.progress_var.set(0)
        
        model = self.model_var.get()
        self.log(f"開始訓練 {model.upper()}...")
        
        self.training_thread = threading.Thread(
            target=self.train_worker, args=(model,), daemon=True
        )
        self.training_thread.start()
    
    def train_worker(self, model_type):
        try:
            start_time = datetime.now()
            
            # 1. 載入數據
            self.log("[1/5] 載入數據...")
            self.progress_var.set(10)
            df = load_btc_data(hf_token=HF_TOKEN)
            if df is None:
                self.log("錯誤: 無法載入數據")
                return
            self.log(f"數據載入完成: {df.shape[0]} 根 K 線")
            
            # 2. 計算指標
            self.log("[2/5] 計算技術指標...")
            self.progress_var.set(30)
            calc = IndicatorCalculator()
            indicators = calc.calculate_all(df)
            self.log(f"指標計算完成: {len(indicators)} 個")
            
            # 3. 構建特徵
            self.log("[3/5] 構建特徵...")
            self.progress_var.set(50)
            
            trend_strength = np.ones(len(df)) * 0.5
            volatility_index = np.ones(len(df)) * 0.5
            direction_confirmation = np.ones(len(df)) * 0.5
            
            from train_models_v2 import ModelTrainerV2
            
            trainer = ModelTrainerV2(df, {})
            X, y = trainer.prepare_features_v2(
                indicators,
                {
                    'trend_strength': trend_strength,
                    'volatility_index': volatility_index,
                    'direction_confirmation': direction_confirmation
                }
            )
            self.log(f"特徵構建完成: {X.shape[1]} 個特徵")
            
            # 4. 訓練
            self.log(f"[4/5] 使用 {model_type.upper()} 訓練模型...")
            self.progress_var.set(70)
            
            if model_type == "lightgbm":
                results = self.train_lightgbm(trainer, X, y)
            elif model_type == "xgboost":
                results = self.train_xgboost(trainer, X, y)
            else:  # lstm
                results = self.train_lstm(X, y)
            
            # 5. 保存
            self.log("[5/5] 保存模型...")
            self.progress_var.set(90)
            if model_type != "lstm":
                trainer.save_models_v2()
            
            # 完成
            self.progress_var.set(100)
            accuracy = results.get('direction_accuracy', results.get('accuracy', 0))
            self.accuracy_label.config(text=f"{accuracy:.2%}")
            
            if accuracy >= 0.70:
                self.accuracy_label.config(foreground="green")
            elif accuracy >= 0.60:
                self.accuracy_label.config(foreground="orange")
            else:
                self.accuracy_label.config(foreground="red")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.time_label.config(text=f"{elapsed:.1f} 秒")
            
            self.log(f"訓練完成! 準確率: {accuracy:.2%}")
            self.log("="*50)
            self.status_label.config(text="完成", foreground="green")
            
        except Exception as e:
            self.log(f"錯誤: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.status_label.config(text="錯誤", foreground="red")
        finally:
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def train_lightgbm(self, trainer, X, y):
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # 分割數據
        test_size = 0.2
        val_size = 0.1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size), shuffle=False
        )
        
        # 正規化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 訓練參數
        params = {
            'n_estimators': self.epochs_var.get(),
            'max_depth': self.depth_var.get(),
            'learning_rate': self.lr_var.get(),
            'num_leaves': self.leaves_var.get(),
            'reg_alpha': self.alpha_var.get(),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        self.log(f"LightGBM 參數: Epochs={params['n_estimators']}, MaxDepth={params['max_depth']}, NumLeaves={params['num_leaves']}, LR={params['learning_rate']}")
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_scaled, y_train['direction'],
            eval_set=[(X_val_scaled, y_val['direction'])],
            callbacks=[lgb.early_stopping(self.early_var.get(), verbose=0)]
        )
        
        # 評估
        y_pred = model.predict(X_test_scaled)
        accuracy = (y_pred == y_test['direction'].values).mean()
        
        self.log(f"測試集準確率: {accuracy:.2%}")
        
        return {'direction_accuracy': accuracy}
    
    def train_xgboost(self, trainer, X, y):
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        test_size = 0.2
        val_size = 0.1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size), shuffle=False
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        params = {
            'n_estimators': self.epochs_var.get(),
            'max_depth': self.depth_var.get(),
            'learning_rate': self.lr_var.get(),
            'subsample': self.subsample_var.get(),
            'random_state': 42
        }
        
        self.log(f"XGBoost 參數: Epochs={params['n_estimators']}, MaxDepth={params['max_depth']}, Subsample={params['subsample']}, LR={params['learning_rate']}")
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_scaled, y_train['direction'],
            eval_set=[(X_val_scaled, y_val['direction'])],
            verbose=False
        )
        
        y_pred = model.predict(X_test_scaled)
        accuracy = (y_pred == y_test['direction'].values).mean()
        
        self.log(f"測試集準確率: {accuracy:.2%}")
        
        return {'direction_accuracy': accuracy}
    
    def train_lstm(self, X, y):
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
        except ImportError:
            self.log("錯誤: TensorFlow 未安裝。請執行: pip install tensorflow")
            return {'accuracy': 0}
        
        self.log(f"LSTM 訓練開始... (Units={self.units_var.get()}, Lookback={self.lookback_var.get()})")
        
        # 正規化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 準備序列數據
        lookback = self.lookback_var.get()
        X_seq = []
        y_seq = []
        
        for i in range(len(X_scaled) - lookback):
            X_seq.append(X_scaled[i:i+lookback])
            y_seq.append(y['direction'].iloc[i+lookback])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 分割
        test_size = int(len(X_seq) * 0.2)
        val_size = int(len(X_seq) * 0.1)
        
        X_train = X_seq[:-test_size-val_size]
        X_val = X_seq[-test_size-val_size:-test_size]
        X_test = X_seq[-test_size:]
        
        y_train = y_seq[:-test_size-val_size]
        y_val = y_seq[-test_size-val_size:-test_size]
        y_test = y_seq[-test_size:]
        
        # 建立模型
        model = Sequential([
            LSTM(self.units_var.get(), activation='relu', input_shape=(lookback, X.shape[1])),
            Dropout(self.dropout_var.get()),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 訓練 (抑制輸出)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs_var.get(),
            batch_size=self.batch_var.get(),
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=self.early_var.get(), restore_best_weights=True)]
        )
        
        # 評估
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        accuracy = (y_pred == y_test).mean()
        
        self.log(f"測試集準確率: {accuracy:.2%}")
        
        return {'accuracy': accuracy}
    
    def stop_training(self):
        self.status_label.config(text="已停止", foreground="orange")
        self.log("訓練已停止")
    
    def clear_logs(self):
        self.log_text.delete(1.0, tk.END)
        self.log("日誌已清除")
    
    def save_params(self):
        params = {
            'model': self.model_var.get(),
            'epochs': self.epochs_var.get(),
            'batch_size': self.batch_var.get(),
            'learning_rate': self.lr_var.get(),
            'early_stopping': self.early_var.get(),
        }
        
        with open('saved_params.json', 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        
        self.log(f"參數已保存至 saved_params.json")
        messagebox.showinfo("成功", "參數已保存")


def main():
    root = tk.Tk()
    gui = AdvancedTrainerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
