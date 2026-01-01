"""
LightGBM + CatBoost 雙模型融合 - PyQt5 GUI + Supply/Demand Zone 檢測與可視化

功能:
  - 訓練模型 (LightGBM + CatBoost)
  - 即時監控訓練進度
  - 模型性能評估
  - 批量預測
  - 模型管理 (保存/載入)
  - Supply/Demand Zone 檢測與可視化 (從 Pine Script 轉寫)
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from threading import Thread
import joblib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QSlider, QProgressBar, QTextEdit, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QDialog, QGroupBox,
    QFormLayout, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QTextCursor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SupplyDemandDetector:
    """基於 Pine Script 邏輯轉寫的 Supply/Demand Zone 檢測模組

    對應原始 Pine Script:
    - aggregationFactor: 聚合倍數 (例如 4 倍當前時間框架)
    - zoneLength: 區間長度 (以 bar 數計)
    - showSupplyZones / showDemandZones: 是否顯示供給/需求區
    - deleteMitigatedZones / deleteBrokenBoxes: 是否刪除已測試或被突破的區域
    """

    def __init__(
        self,
        aggregation_factor: int = 4,
        zone_length: int = 50,
        show_supply_zones: bool = True,
        show_demand_zones: bool = True,
        delete_mitigated_zones: bool = False,
        delete_broken_zones: bool = True,
    ):
        self.aggregation_factor = aggregation_factor
        self.zone_length = zone_length
        self.show_supply_zones = show_supply_zones
        self.show_demand_zones = show_demand_zones
        self.delete_mitigated_zones = delete_mitigated_zones
        self.delete_broken_zones = delete_broken_zones

    def detect_zones(self, df: pd.DataFrame):
        """在 OHLC 資料上偵測 Supply / Demand 區域。

        這裡假設 df 有欄位: ['open', 'high', 'low', 'close']，index 是時間。
        我們會模擬 Pine Script 中的「聚合 K 線」邏輯：
        - 使用 aggregation_factor * 原始時間框架 的視角，構造虛擬聚合 K 線
        - 在聚合 K 線轉多頭/空頭時，紀錄前一段區間，形成 Supply / Demand 區

        返回:
            zones: dict
                {
                    'supply': list of dicts: { 'start_idx', 'end_idx', 'high', 'low' }
                    'demand': list of dicts: { 'start_idx', 'end_idx', 'high', 'low' }
                }
        """
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)

        # 聚合組相關狀態
        group_start_idx = None
        agg_open = agg_high = agg_low = agg_close = None

        prev_supply_low = None
        prev_supply_high = None
        prev_supply_start = None

        prev_demand_low = None
        prev_demand_high = None
        prev_demand_start = None

        supply_zone_used = False
        demand_zone_used = False

        supply_zones = []
        demand_zones = []

        # 我們不用實際時間去對齊 UTC-5，只用「每 aggregation_factor 根 bar 視為一組」簡化實作，
        # 邏輯上等價於在 Tv 上用更高級別時間框架觀察。
        for i in range(n):
            # 判斷是否進入新聚合組
            if group_start_idx is None:
                # 初始化第一組
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]
                continue

            # 聚合更新
            agg_high = max(agg_high, h[i])
            agg_low = min(agg_low, l[i])
            agg_close = c[i]

            bars_in_group = i - group_start_idx + 1
            is_new_group = bars_in_group >= self.aggregation_factor

            if is_new_group:
                # 一個聚合 K 線完成，根據其多空性決定 Supply/Demand 候選區
                is_bullish = agg_close >= agg_open

                if is_bullish:
                    # 多頭聚合 K 線：根據 Pine 邏輯，更新「上一個可能的 supply 區」，
                    # 並檢查是否形成 demand 區
                    prev_supply_low = agg_low
                    prev_supply_high = agg_high
                    prev_supply_start = group_start_idx
                    supply_zone_used = False

                    # 檢查是否突破前一 demand 高點形成 demand 區
                    if (
                        self.show_demand_zones
                        and prev_demand_high is not None
                        and agg_close > prev_demand_high
                        and not demand_zone_used
                    ):
                        zone_start = prev_demand_start
                        zone_end = min(prev_demand_start + self.zone_length, n - 1)
                        demand_zones.append(
                            {
                                'start_idx': zone_start,
                                'end_idx': zone_end,
                                'high': prev_demand_high,
                                'low': prev_demand_low,
                            }
                        )
                        demand_zone_used = True

                else:
                    # 空頭聚合 K 線：更新「上一個可能的 demand 區」，
                    # 並檢查是否跌破前一 supply 低點形成 supply 區
                    prev_demand_low = agg_low
                    prev_demand_high = agg_high
                    prev_demand_start = group_start_idx
                    demand_zone_used = False

                    if (
                        self.show_supply_zones
                        and prev_supply_low is not None
                        and agg_close < prev_supply_low
                        and not supply_zone_used
                    ):
                        zone_start = prev_supply_start
                        zone_end = min(prev_supply_start + self.zone_length, n - 1)
                        supply_zones.append(
                            {
                                'start_idx': zone_start,
                                'end_idx': zone_end,
                                'high': prev_supply_high,
                                'low': prev_supply_low,
                            }
                        )
                        supply_zone_used = True

                # 新聚合組開始
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]

        return {
            'supply': supply_zones,
            'demand': demand_zones,
        }


class KlineCanvas(FigureCanvas):
    """用於在 PyQt5 中嵌入 K 線與 Supply/Demand Zone 的 Matplotlib 畫布"""

    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def plot_kline_with_zones(self, df: pd.DataFrame, zones: dict):
        self.ax.clear()

        if df is None or len(df) == 0:
            self.ax.text(0.5, 0.5, "尚未載入數據", ha='center', va='center', transform=self.ax.transAxes)
            self.draw()
            return

        # 畫基本 K 線 (簡化版 OHLC 條)
        x = np.arange(len(df))
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values

        # 上漲 K 線
        up = c >= o
        down = ~up

        # 畫上下影線
        self.ax.vlines(x, l, h, color='black', linewidth=0.5, alpha=0.6)
        # 實體
        self.ax.vlines(x[up], o[up], c[up], color='green', linewidth=4)
        self.ax.vlines(x[down], c[down], o[down], color='red', linewidth=4)

        # 畫 Supply/Demand Zones
        if zones is not None:
            # Supply
            for z in zones.get('supply', []):
                self.ax.axhspan(z['low'], z['high'], xmin=z['start_idx']/len(df), xmax=z['end_idx']/len(df),
                                facecolor=(1, 0, 0, 0.2), edgecolor='red', linewidth=1)
            # Demand
            for z in zones.get('demand', []):
                self.ax.axhspan(z['low'], z['high'], xmin=z['start_idx']/len(df), xmax=z['end_idx']/len(df),
                                facecolor=(0, 1, 0, 0.2), edgecolor='green', linewidth=1)

        self.ax.set_title("K 線與 Supply/Demand 區域")
        self.ax.set_xlabel("Bar Index")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, alpha=0.2)
        self.fig.tight_layout()
        self.draw()


class TrainingWorker(QThread):
    """後台訓練執行緒"""
    progress = pyqtSignal(str)  # 進度訊息
    finished = pyqtSignal(dict)  # 訓練完成結果
    error = pyqtSignal(str)  # 錯誤訊息
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, config):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        self.scaler = StandardScaler()
        
    def run(self):
        try:
            self.progress.emit("開始訓練...\n")
            
            # 特徵縮放
            self.progress.emit("[1/5] 特徵縮放...")
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_val_scaled = self.scaler.transform(self.X_val)
            X_test_scaled = self.scaler.transform(self.X_test)
            self.progress.emit("✓ 特徵縮放完成\n")
            
            # 訓練 LightGBM
            self.progress.emit("[2/5] 訓練 LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=self.config['lgb_estimators'],
                max_depth=self.config['lgb_depth'],
                num_leaves=self.config['lgb_leaves'],
                learning_rate=self.config['lgb_lr'],
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            lgb_model.fit(
                X_train_scaled, self.y_train,
                eval_set=[(X_val_scaled, self.y_val)],
                eval_metric='binary_logloss',
                callbacks=[
                    lgb.early_stopping(10, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            lgb_val_pred = lgb_model.predict(X_val_scaled)
            lgb_val_acc = accuracy_score(self.y_val, lgb_val_pred)
            lgb_test_pred = lgb_model.predict(X_test_scaled)
            lgb_test_acc = accuracy_score(self.y_test, lgb_test_pred)
            
            self.progress.emit(f"✓ LightGBM 訓練完成 (驗證: {lgb_val_acc:.4f}, 測試: {lgb_test_acc:.4f})\n")
            
            # 訓練 CatBoost
            self.progress.emit("[3/5] 訓練 CatBoost...")
            cb_model = CatBoostClassifier(
                iterations=self.config['cb_iterations'],
                max_depth=self.config['cb_depth'],
                learning_rate=self.config['cb_lr'],
                subsample=0.8,
                random_state=42,
                verbose=0
            )
            
            cb_model.fit(
                X_train_scaled, self.y_train,
                eval_set=[(X_val_scaled, self.y_val)],
                verbose=False
            )
            
            cb_val_pred = cb_model.predict(X_val_scaled)
            cb_val_acc = accuracy_score(self.y_val, cb_val_pred)
            cb_test_pred = cb_model.predict(X_test_scaled)
            cb_test_acc = accuracy_score(self.y_test, cb_test_pred)
            
            self.progress.emit(f"✓ CatBoost 訓練完成 (驗證: {cb_val_acc:.4f}, 測試: {cb_test_acc:.4f})\n")
            
            # 融合預測
            self.progress.emit("[4/5] 計算融合預測...")
            lgb_weight = self.config['lgb_weight']
            cb_weight = self.config['cb_weight']
            total = lgb_weight + cb_weight
            lgb_weight /= total
            cb_weight /= total
            
            lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
            cb_proba = cb_model.predict_proba(X_test_scaled)[:, 1]
            ensemble_proba = lgb_weight * lgb_proba + cb_weight * cb_proba
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            
            ensemble_acc = accuracy_score(self.y_test, ensemble_pred)
            ensemble_precision = precision_score(self.y_test, ensemble_pred)
            ensemble_recall = recall_score(self.y_test, ensemble_pred)
            ensemble_f1 = f1_score(self.y_test, ensemble_pred)
            
            self.progress.emit(f"✓ 融合完成 (準確率: {ensemble_acc:.4f})\n")
            
            # 保存模型
            self.progress.emit("[5/5] 保存模型...")
            model_dir = Path('trained_models')
            model_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lgb_path = model_dir / f"lightgbm_{timestamp}.pkl"
            cb_path = model_dir / f"catboost_{timestamp}.pkl"
            scaler_path = model_dir / f"scaler_{timestamp}.pkl"
            
            joblib.dump(lgb_model, lgb_path)
            joblib.dump(cb_model, cb_path)
            joblib.dump(self.scaler, scaler_path)
            
            self.progress.emit(f"✓ 模型已保存\n")
            
            # 返回結果
            results = {
                'lgb_val_acc': lgb_val_acc,
                'lgb_test_acc': lgb_test_acc,
                'cb_val_acc': cb_val_acc,
                'cb_test_acc': cb_test_acc,
                'ensemble_acc': ensemble_acc,
                'ensemble_precision': ensemble_precision,
                'ensemble_recall': ensemble_recall,
                'ensemble_f1': ensemble_f1,
                'lgb_path': str(lgb_path),
                'cb_path': str(cb_path),
                'scaler_path': str(scaler_path),
                'lgb_weight': lgb_weight,
                'cb_weight': cb_weight
            }
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(f"訓練錯誤: {str(e)}")


class PredictionWorker(QThread):
    """預測執行緒"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(tuple)  # (predictions, ensemble_proba)
    error = pyqtSignal(str)
    
    def __init__(self, X, lgb_model, cb_model, scaler):
        super().__init__()
        self.X = X
        self.lgb_model = lgb_model
        self.cb_model = cb_model
        self.scaler = scaler
    
    def run(self):
        try:
            self.progress.emit("正在預測...")
            
            X_scaled = self.scaler.transform(self.X)
            lgb_proba = self.lgb_model.predict_proba(X_scaled)[:, 1]
            cb_proba = self.cb_model.predict_proba(X_scaled)[:, 1]
            
            ensemble_proba = 0.5 * lgb_proba + 0.5 * cb_proba
            predictions = (ensemble_proba >= 0.5).astype(int)
            
            self.progress.emit("預測完成")
            self.finished.emit((predictions, ensemble_proba))
            
        except Exception as e:
            self.error.emit(f"預測錯誤: {str(e)}")


class ModelEnsembleGUI(QMainWindow):
    """模型融合 GUI 主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LightGBM + CatBoost 雙模型融合系統")
        self.setGeometry(100, 100, 1400, 900)
        
        # 訓練數據
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # 原始 K 線數據（供可視化與 Supply/Demand 檢測使用）
        self.df_kline = None
        self.zones = None
        self.sd_detector = SupplyDemandDetector()
        
        # 模型
        self.scaler = None
        self.lgb_model = None
        self.cb_model = None
        
        # 執行緒
        self.training_worker = None
        self.prediction_worker = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """建立 UI"""
        # 主視窗
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        
        # 標題
        title = QLabel("LightGBM + CatBoost 雙模型融合系統 + Supply/Demand Zones")
        title.setFont(QFont("微軟正黑體", 16, QFont.Bold))
        main_layout.addWidget(title)
        
        # 標籤頁
        tabs = QTabWidget()
        
        # [標籤 1] 數據載入
        tabs.addTab(self.create_data_tab(), "數據載入")
        
        # [標籤 2] 模型訓練
        tabs.addTab(self.create_training_tab(), "模型訓練")
        
        # [標籤 3] 模型評估
        tabs.addTab(self.create_evaluation_tab(), "模型評估")
        
        # [標籤 4] 預測
        tabs.addTab(self.create_prediction_tab(), "預測")

        # [標籤 5] Supply/Demand 可視化
        tabs.addTab(self.create_supply_demand_tab(), "Supply/Demand 可視化")
        
        main_layout.addWidget(tabs)
        main_widget.setLayout(main_layout)
        
    def create_data_tab(self):
        """數據載入標籤頁"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 按鈕
        btn_load = QPushButton("從檔案載入數據 (CSV/Parquet)")
        btn_load.clicked.connect(self.load_data)
        layout.addWidget(btn_load)
        
        btn_load_hf = QPushButton("從 Hugging Face 載入 BTC 數據")
        btn_load_hf.clicked.connect(self.load_data_hf)
        layout.addWidget(btn_load_hf)
        
        # 狀態文本
        self.data_status = QTextEdit()
        self.data_status.setReadOnly(True)
        layout.addWidget(QLabel("狀態信息:"))
        layout.addWidget(self.data_status)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_training_tab(self):
        """模型訓練標籤頁"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # LightGBM 參數
        lgb_group = QGroupBox("LightGBM 參數")
        lgb_form = QFormLayout()
        
        self.lgb_estimators = QSpinBox()
        self.lgb_estimators.setValue(200)
        self.lgb_estimators.setRange(50, 1000)
        lgb_form.addRow("樹數量:", self.lgb_estimators)
        
        self.lgb_depth = QSpinBox()
        self.lgb_depth.setValue(11)
        self.lgb_depth.setRange(3, 20)
        lgb_form.addRow("最大深度:", self.lgb_depth)
        
        self.lgb_leaves = QSpinBox()
        self.lgb_leaves.setValue(63)
        self.lgb_leaves.setRange(31, 255)
        lgb_form.addRow("葉子數:", self.lgb_leaves)
        
        self.lgb_lr = QDoubleSpinBox()
        self.lgb_lr.setValue(0.05)
        self.lgb_lr.setRange(0.001, 0.5)
        self.lgb_lr.setSingleStep(0.01)
        lgb_form.addRow("學習率:", self.lgb_lr)
        
        lgb_group.setLayout(lgb_form)
        layout.addWidget(lgb_group)
        
        # CatBoost 參數
        cb_group = QGroupBox("CatBoost 參數")
        cb_form = QFormLayout()
        
        self.cb_iterations = QSpinBox()
        self.cb_iterations.setValue(200)
        self.cb_iterations.setRange(50, 1000)
        cb_form.addRow("迭代次數:", self.cb_iterations)
        
        self.cb_depth = QSpinBox()
        self.cb_depth.setValue(8)
        self.cb_depth.setRange(3, 16)
        cb_form.addRow("最大深度:", self.cb_depth)
        
        self.cb_lr = QDoubleSpinBox()
        self.cb_lr.setValue(0.1)
        self.cb_lr.setRange(0.001, 0.5)
        self.cb_lr.setSingleStep(0.01)
        cb_form.addRow("學習率:", self.cb_lr)
        
        cb_group.setLayout(cb_form)
        layout.addWidget(cb_group)
        
        # 融合權重
        weight_group = QGroupBox("融合權重")
        weight_form = QFormLayout()
        
        self.lgb_weight = QDoubleSpinBox()
        self.lgb_weight.setValue(0.5)
        self.lgb_weight.setRange(0.0, 1.0)
        self.lgb_weight.setSingleStep(0.1)
        weight_form.addRow("LightGBM 權重:", self.lgb_weight)
        
        self.cb_weight = QDoubleSpinBox()
        self.cb_weight.setValue(0.5)
        self.cb_weight.setRange(0.0, 1.0)
        self.cb_weight.setSingleStep(0.1)
        weight_form.addRow("CatBoost 權重:", self.cb_weight)
        
        weight_group.setLayout(weight_form)
        layout.addWidget(weight_group)
        
        # 訓練按鈕
        btn_train = QPushButton("開始訓練")
        btn_train.setFont(QFont("微軟正黑體", 12, QFont.Bold))
        btn_train.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        btn_train.clicked.connect(self.start_training)
        layout.addWidget(btn_train)
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 訓練日誌
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        layout.addWidget(QLabel("訓練日誌:"))
        layout.addWidget(self.training_log)
        
        widget.setLayout(layout)
        return widget
    
    def create_evaluation_tab(self):
        """模型評估標籤頁"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 評估結果表
        self.eval_table = QTableWidget()
        self.eval_table.setColumnCount(2)
        self.eval_table.setHorizontalHeaderLabels(["指標", "數值"])
        self.eval_table.setMaximumHeight(400)
        
        layout.addWidget(QLabel("模型性能評估"))
        layout.addWidget(self.eval_table)
        
        # 詳細報告
        self.eval_report = QTextEdit()
        self.eval_report.setReadOnly(True)
        layout.addWidget(QLabel("詳細報告:"))
        layout.addWidget(self.eval_report)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_prediction_tab(self):
        """預測標籤頁"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 模型選擇
        model_group = QGroupBox("模型選擇")
        model_form = QFormLayout()
        
        btn_load_model = QPushButton("載入訓練好的模型")
        btn_load_model.clicked.connect(self.load_models)
        model_form.addRow(btn_load_model)
        
        self.model_status = QLabel("未載入模型")
        model_form.addRow("模型狀態:", self.model_status)
        
        model_group.setLayout(model_form)
        layout.addWidget(model_group)
        
        # 預測方法
        pred_group = QGroupBox("預測方法")
        pred_form = QFormLayout()
        
        btn_pred_hf = QPushButton("從 Hugging Face 預測")
        btn_pred_hf.clicked.connect(self.predict_from_hf)
        pred_form.addRow(btn_pred_hf)
        
        btn_pred_file = QPushButton("從檔案預測 (CSV)")
        btn_pred_file.clicked.connect(self.predict_from_file)
        pred_form.addRow(btn_pred_file)
        
        pred_group.setLayout(pred_form)
        layout.addWidget(pred_group)
        
        # 預測結果
        self.prediction_result = QTextEdit()
        self.prediction_result.setReadOnly(True)
        layout.addWidget(QLabel("預測結果:"))
        layout.addWidget(self.prediction_result)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_supply_demand_tab(self):
        """Supply/Demand 可視化標籤頁"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 參數設定
        param_group = QGroupBox("Supply/Demand 檢測參數")
        param_form = QFormLayout()

        self.sd_agg_factor = QSpinBox()
        self.sd_agg_factor.setRange(1, 20)
        self.sd_agg_factor.setValue(4)
        param_form.addRow("Aggregation Factor:", self.sd_agg_factor)

        self.sd_zone_length = QSpinBox()
        self.sd_zone_length.setRange(10, 500)
        self.sd_zone_length.setValue(50)
        param_form.addRow("Zone Length (bars):", self.sd_zone_length)

        self.sd_show_supply = QCheckBox("顯示 Supply Zones")
        self.sd_show_supply.setChecked(True)
        param_form.addRow(self.sd_show_supply)

        self.sd_show_demand = QCheckBox("顯示 Demand Zones")
        self.sd_show_demand.setChecked(True)
        param_form.addRow(self.sd_show_demand)

        self.sd_delete_mitigated = QCheckBox("刪除已測試 Zones")
        self.sd_delete_mitigated.setChecked(False)
        param_form.addRow(self.sd_delete_mitigated)

        self.sd_delete_broken = QCheckBox("刪除被突破 Zones")
        self.sd_delete_broken.setChecked(True)
        param_form.addRow(self.sd_delete_broken)

        btn_run_sd = QPushButton("重新檢測並繪製 Zones")
        btn_run_sd.clicked.connect(self.run_supply_demand_detection)
        param_form.addRow(btn_run_sd)

        param_group.setLayout(param_form)
        layout.addWidget(param_group)

        # K 線畫布
        self.kline_canvas = KlineCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.kline_canvas)

        widget.setLayout(layout)
        return widget
    
    def load_data(self):
        """從 CSV/Parquet 載入數據，若包含 OHLC 欄位則同時保存為 K 線資料"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇數據檔案", "", "CSV Files (*.csv);;Parquet Files (*.parquet)"
        )
        
        if not file_path:
            return
        
        try:
            self.data_status.setText("正在載入數據...")
            QApplication.processEvents()
            
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # 如果資料包含 OHLC 欄位，則當作 K 線來源
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                self.df_kline = df.copy()

            # 簡單特徵工程 (假設有 features 和 target)
            if 'direction' in df.columns:
                y = df['direction'].values
                X = df.drop(['direction'], axis=1).values
            else:
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            
            # 分割數據
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.125, shuffle=False
            )
            
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test
            
            status = f"數據載入成功!\n"
            status += f"訓練集: {X_train.shape}\n"
            status += f"驗證集: {X_val.shape}\n"
            status += f"測試集: {X_test.shape}\n"
            status += f"特徵數: {X_train.shape[1]}"
            
            if self.df_kline is not None:
                status += f"\nK 線資料可用, 行數: {len(self.df_kline)}"
            
            self.data_status.setText(status)
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"載入數據失敗: {str(e)}")
    
    def load_data_hf(self):
        """從 Hugging Face 載入 BTC 數據"""
        try:
            self.data_status.setText("正在從 Hugging Face 載入...")
            QApplication.processEvents()
            
            from data import load_btc_data
            from indicators import IndicatorCalculator
            from train_models_v2 import ModelTrainerV2
            from config import HF_TOKEN
            
            # 載入數據
            df = load_btc_data(hf_token=HF_TOKEN)
            self.df_kline = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # 計算指標
            calc = IndicatorCalculator()
            indicators = calc.calculate_all(df)
            
            # 構建特徵
            trend_strength = np.ones(len(df)) * 0.5
            volatility_index = np.ones(len(df)) * 0.5
            direction_confirmation = np.ones(len(df)) * 0.5
            
            trainer = ModelTrainerV2(df, {})
            X, y = trainer.prepare_features_v2(
                indicators,
                {
                    'trend_strength': trend_strength,
                    'volatility_index': volatility_index,
                    'direction_confirmation': direction_confirmation
                }
            )
            
            # 分割數據
            X_train, X_test, y_train, y_test = train_test_split(
                X, y['direction'].values, test_size=0.2, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.125, shuffle=False
            )
            
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test
            
            status = f"BTC 數據載入成功!\n"
            status += f"訓練集: {X_train.shape}\n"
            status += f"驗證集: {X_val.shape}\n"
            status += f"測試集: {X_test.shape}\n"
            status += f"特徵數: {X_train.shape[1]}\n"
            status += f"K 線資料可用, 行數: {len(self.df_kline)}"
            
            self.data_status.setText(status)
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"載入數據失敗: {str(e)}")
    
    def start_training(self):
        """開始訓練"""
        if self.X_train is None:
            QMessageBox.warning(self, "警告", "請先載入數據")
            return
        
        config = {
            'lgb_estimators': self.lgb_estimators.value(),
            'lgb_depth': self.lgb_depth.value(),
            'lgb_leaves': self.lgb_leaves.value(),
            'lgb_lr': self.lgb_lr.value(),
            'cb_iterations': self.cb_iterations.value(),
            'cb_depth': self.cb_depth.value(),
            'cb_lr': self.cb_lr.value(),
            'lgb_weight': self.lgb_weight.value(),
            'cb_weight': self.cb_weight.value()
        }
        
        # 開始訓練執行緒
        self.training_worker = TrainingWorker(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
            config
        )
        
        self.training_worker.progress.connect(self.update_training_log)
        self.training_worker.finished.connect(self.training_finished)
        self.training_worker.error.connect(self.training_error)
        self.training_worker.start()
        
        self.progress_bar.setValue(50)
    
    def update_training_log(self, message):
        """更新訓練日誌"""
        cursor = self.training_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message + "\n")
        self.training_log.setTextCursor(cursor)
        self.training_log.ensureCursorVisible()
    
    def training_finished(self, results):
        """訓練完成回調"""
        self.progress_bar.setValue(100)
        
        # 更新評估表
        self.eval_table.setRowCount(8)
        
        metrics = [
            ("LightGBM 驗證準確率", f"{results['lgb_val_acc']:.4f}"),
            ("LightGBM 測試準確率", f"{results['lgb_test_acc']:.4f}"),
            ("CatBoost 驗證準確率", f"{results['cb_val_acc']:.4f}"),
            ("CatBoost 測試準確率", f"{results['cb_test_acc']:.4f}"),
            ("融合模型準確率", f"{results['ensemble_acc']:.4f}"),
            ("Precision", f"{results['ensemble_precision']:.4f}"),
            ("Recall", f"{results['ensemble_recall']:.4f}"),
            ("F1 Score", f"{results['ensemble_f1']:.4f}")
        ]
        
        for i, (metric, value) in enumerate(metrics):
            self.eval_table.setItem(i, 0, QTableWidgetItem(metric))
            self.eval_table.setItem(i, 1, QTableWidgetItem(value))
        
        # 詳細報告
        report = f"""
模型訓練完成!

LightGBM:
  驗證準確率: {results['lgb_val_acc']:.4f}
  測試準確率: {results['lgb_test_acc']:.4f}

CatBoost:
  驗證準確率: {results['cb_val_acc']:.4f}
  測試準確率: {results['cb_test_acc']:.4f}

融合模型 (LightGBM {results['lgb_weight']:.1%} + CatBoost {results['cb_weight']:.1%}):
  準確率: {results['ensemble_acc']:.4f}
  Precision: {results['ensemble_precision']:.4f}
  Recall: {results['ensemble_recall']:.4f}
  F1 Score: {results['ensemble_f1']:.4f}

模型已保存:
  LightGBM: {results['lgb_path']}
  CatBoost: {results['cb_path']}
  Scaler: {results['scaler_path']}
        """
        
        self.eval_report.setText(report)
        
        QMessageBox.information(self, "成功", "模型訓練完成!")
    
    def training_error(self, error_msg):
        """訓練錯誤回調"""
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "錯誤", error_msg)
    
    def load_models(self):
        """載入模型"""
        lgb_path, _ = QFileDialog.getOpenFileName(
            self, "選擇 LightGBM 模型", "trained_models", "PKL Files (*.pkl)"
        )
        
        if not lgb_path:
            return
        
        cb_path, _ = QFileDialog.getOpenFileName(
            self, "選擇 CatBoost 模型", "trained_models", "PKL Files (*.pkl)"
        )
        
        if not cb_path:
            return
        
        scaler_path, _ = QFileDialog.getOpenFileName(
            self, "選擇 Scaler", "trained_models", "PKL Files (*.pkl)"
        )
        
        if not scaler_path:
            return
        
        try:
            self.lgb_model = joblib.load(lgb_path)
            self.cb_model = joblib.load(cb_path)
            self.scaler = joblib.load(scaler_path)
            
            self.model_status.setText("✓ 模型已載入")
            self.model_status.setStyleSheet("color: green;")
            
            self.prediction_result.setText("模型已載入,可進行預測")
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"載入模型失敗: {str(e)}")
    
    def predict_from_hf(self):
        """從 Hugging Face 數據預測"""
        if self.lgb_model is None or self.cb_model is None:
            QMessageBox.warning(self, "警告", "請先載入模型")
            return
        
        try:
            self.prediction_result.setText("正在從 Hugging Face 載入數據...")
            QApplication.processEvents()
            
            from data import load_btc_data
            from indicators import IndicatorCalculator
            from train_models_v2 import ModelTrainerV2
            from config import HF_TOKEN
            
            # 載入數據
            df = load_btc_data(hf_token=HF_TOKEN)
            self.df_kline = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # 計算指標
            calc = IndicatorCalculator()
            indicators = calc.calculate_all(df)
            
            # 構建特徵
            trend_strength = np.ones(len(df)) * 0.5
            volatility_index = np.ones(len(df)) * 0.5
            direction_confirmation = np.ones(len(df)) * 0.5
            
            trainer = ModelTrainerV2(df, {})
            X, y = trainer.prepare_features_v2(
                indicators,
                {
                    'trend_strength': trend_strength,
                    'volatility_index': volatility_index,
                    'direction_confirmation': direction_confirmation
                }
            )
            
            # 預測
            self.prediction_worker = PredictionWorker(
                X.values, self.lgb_model, self.cb_model, self.scaler
            )
            
            self.prediction_worker.finished.connect(self.prediction_finished)
            self.prediction_worker.error.connect(self.prediction_error)
            self.prediction_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"預測失敗: {str(e)}")
    
    def predict_from_file(self):
        """從檔案預測"""
        if self.lgb_model is None or self.cb_model is None:
            QMessageBox.warning(self, "警告", "請先載入模型")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇預測數據", "", "CSV Files (*.csv);;Parquet Files (*.parquet)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # 若包含 OHLC 欄位，更新 K 線資料
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                self.df_kline = df[['open', 'high', 'low', 'close']].copy()
            
            X = df.values if 'direction' not in df.columns else df.drop(['direction'], axis=1).values
            
            # 預測
            self.prediction_worker = PredictionWorker(
                X, self.lgb_model, self.cb_model, self.scaler
            )
            
            self.prediction_worker.finished.connect(self.prediction_finished)
            self.prediction_worker.error.connect(self.prediction_error)
            self.prediction_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"預測失敗: {str(e)}")
    
    def prediction_finished(self, results):
        """預測完成回調"""
        predictions, ensemble_proba = results
        
        # 顯示結果
        result_text = f"預測完成!\n"
        result_text += f"樣本数: {len(predictions)}\n"
        result_text += f"上升 (1): {(predictions == 1).sum()} ({(predictions == 1).sum() / len(predictions) * 100:.2f}%)\n"
        result_text += f"下降 (0): {(predictions == 0).sum()} ({(predictions == 0).sum() / len(predictions) * 100:.2f}%)\n\n"
        result_text += "前 30 個預測結果\n"
        result_text += "-" * 60 + "\n"
        result_text += "索引\t預測 (0/1)\t概率 (%)\n"
        result_text += "-" * 60 + "\n"
        
        for i in range(min(30, len(predictions))):
            result_text += f"{i}\t{predictions[i]}\t{ensemble_proba[i]*100:.2f}\n"
        
        self.prediction_result.setText(result_text)
        
        # 選項：保存結果
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存預測結果", "", "CSV Files (*.csv)"
        )
        
        if save_path:
            result_df = pd.DataFrame({
                'prediction': predictions,
                'probability': ensemble_proba
            })
            result_df.to_csv(save_path, index=False)
            QMessageBox.information(self, "成功", f"預測結果已保存到 {save_path}")
    
    def prediction_error(self, error_msg):
        """預測錯誤回調"""
        QMessageBox.critical(self, "錯誤", error_msg)

    def run_supply_demand_detection(self):
        """依據目前參數重新檢測 Supply/Demand Zones 並更新圖表"""
        if self.df_kline is None or len(self.df_kline) == 0:
            QMessageBox.warning(self, "警告", "尚未載入包含 OHLC 的 K 線數據")
            return

        # 更新檢測器參數
        self.sd_detector = SupplyDemandDetector(
            aggregation_factor=self.sd_agg_factor.value(),
            zone_length=self.sd_zone_length.value(),
            show_supply_zones=self.sd_show_supply.isChecked(),
            show_demand_zones=self.sd_show_demand.isChecked(),
            delete_mitigated_zones=self.sd_delete_mitigated.isChecked(),
            delete_broken_zones=self.sd_delete_broken.isChecked(),
        )

        self.zones = self.sd_detector.detect_zones(self.df_kline)
        self.kline_canvas.plot_kline_with_zones(self.df_kline, self.zones)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    app = QApplication(sys.argv)
    window = ModelEnsembleGUI()
    window.show()
    sys.exit(app.exec_())
