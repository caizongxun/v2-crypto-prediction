import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import sys
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QCheckBox, QTabWidget,
    QTextEdit, QFileDialog, QMessageBox, QProgressBar, QGroupBox,
    QGridLayout, QSplitter, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


# 支援的幣種列表 (23種)
SUPPORTED_SYMBOLS = [
    'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT',
    'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
    'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
    'UNIUSDT', 'XRPUSDT'
]

TIMEFRAMES = ['15m', '1h']
HF_DATASET_ID = 'zongowo111/v2-crypto-ohlcv-data'
HF_DATASET_PATH = 'klines'


class DataLoaderWorker(QThread):
    """資料加載工作執行緒 - 從HuggingFace下載OHLCV資料"""
    progress_signal = pyqtSignal(str)
    completed_signal = pyqtSignal(pd.DataFrame)
    error_signal = pyqtSignal(str)

    def __init__(self, symbol: str, timeframe: str):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe

    def run(self):
        """執行資料加載"""
        try:
            self.progress_signal.emit(f'正在從 Hugging Face 下載 {self.symbol} {self.timeframe} 資料...')
            
            df = self._download_from_huggingface()
            
            if df is None or df.empty:
                raise ValueError(f'無法獲取 {self.symbol} {self.timeframe} 資料')
            
            self.progress_signal.emit(f'正在驗證和處理 {self.symbol} 資料...')
            df = self._validate_and_process(df)
            
            self.progress_signal.emit(f'已加載 {len(df)} 筆 {self.symbol} {self.timeframe} 資料')
            self.completed_signal.emit(df)
            
        except Exception as e:
            self.error_signal.emit(f'加載資料失敗: {str(e)}')

    def _download_from_huggingface(self) -> Optional[pd.DataFrame]:
        """
        從 HuggingFace 下載資料
        
        資料集結構:
        klines/
            ├── BTCUSDT/
            │   ├── BTC_15m.parquet
            │   └── BTC_1h.parquet
            ├── ETHUSDT/
            │   ├── ETH_15m.parquet
            │   └── ETH_1h.parquet
            └── ...
        """
        try:
            # 構建檔案路徑
            symbol_without_usdt = self.symbol.replace('USDT', '')
            filename = f'{symbol_without_usdt}_{self.timeframe}.parquet'
            filepath = f'{HF_DATASET_PATH}/{self.symbol}/{filename}'
            
            self.progress_signal.emit(f'正在下載檔案: {filepath}')
            
            # 從 HuggingFace Hub 下載檔案
            parquet_path = hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=filepath,
                repo_type='dataset',
                cache_dir=None  # 使用預設快取目錄
            )
            
            self.progress_signal.emit(f'正在讀取 {filepath}...')
            
            # 讀取 parquet 檔案
            df = pd.read_parquet(parquet_path)
            
            return df
            
        except Exception as e:
            raise Exception(f'HuggingFace 下載失敗 ({self.symbol}/{self.timeframe}): {str(e)}')

    def _validate_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """驗證並處理資料"""
        # 定義必要欄位 (支援多種欄位名稱格式)
        required_cols_variants = {
            'timestamp': ['timestamp', 'time', 'datetime', 'date'],
            'open': ['open', 'o'],
            'high': ['high', 'h'],
            'low': ['low', 'l'],
            'close': ['close', 'c'],
            'volume': ['volume', 'vol', 'v']
        }
        
        # 標準化欄位名稱
        df_processed = df.copy()
        col_mapping = {}
        
        for standard_name, variants in required_cols_variants.items():
            found = False
            for variant in variants:
                if variant in df_processed.columns:
                    col_mapping[variant] = standard_name
                    found = True
                    break
            
            if not found:
                raise ValueError(f'缺少必要欄位: {standard_name}')
        
        # 重新命名欄位
        df_processed = df_processed.rename(columns=col_mapping)
        
        # 確保只保留必要欄位
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_processed = df_processed[required_cols].copy()
        
        # 轉換資料型別
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # 轉換 timestamp 為 datetime
        if df_processed['timestamp'].dtype == 'object':
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
        
        # 移除無效資料
        df_processed = df_processed.dropna()
        
        # 排序
        df_processed = df_processed.sort_values('timestamp').reset_index(drop=True)
        
        # 驗證資料完整性
        if len(df_processed) == 0:
            raise ValueError('處理後無有效資料')
        
        # 檢查高、低、開、收的合理性
        invalid_rows = (df_processed['high'] < df_processed['low']) | \
                      (df_processed['close'] > df_processed['high']) | \
                      (df_processed['close'] < df_processed['low'])
        
        if invalid_rows.sum() > 0:
            print(f'警告: 發現 {invalid_rows.sum()} 筆異常資料，已移除')
            df_processed = df_processed[~invalid_rows].reset_index(drop=True)
        
        return df_processed


class TrainingWorker(QThread):
    """模型訓練工作執行緒"""
    progress_signal = pyqtSignal(str)
    completed_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        """執行訓練"""
        try:
            self.progress_signal.emit('開始訓練模型...')
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)
            
            # LightGBM 訓練
            self.progress_signal.emit('訓練 LightGBM 模型...')
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train_scaled, self.y_train)
            lgb_pred = lgb_model.predict(X_test_scaled)
            lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
            
            # CatBoost 訓練
            self.progress_signal.emit('訓練 CatBoost 模型...')
            cb_model = CatBoostClassifier(
                iterations=100,
                max_depth=7,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
            cb_model.fit(X_train_scaled, self.y_train)
            cb_pred = cb_model.predict(X_test_scaled)
            cb_proba = cb_model.predict_proba(X_test_scaled)[:, 1]
            
            # 計算準確度
            lgb_acc = (lgb_pred == self.y_test).sum() / len(self.y_test)
            cb_acc = (cb_pred == self.y_test).sum() / len(self.y_test)
            
            # Ensemble 預測
            ensemble_pred = ((lgb_proba + cb_proba) / 2 > 0.5).astype(int)
            ensemble_acc = (ensemble_pred == self.y_test).sum() / len(self.y_test)
            
            results = {
                'lgb_model': lgb_model,
                'cb_model': cb_model,
                'scaler': scaler,
                'lgb_acc': lgb_acc,
                'cb_acc': cb_acc,
                'ensemble_acc': ensemble_acc,
                'lgb_pred': lgb_pred,
                'cb_pred': cb_pred,
                'ensemble_pred': ensemble_pred,
                'y_test': self.y_test
            }
            
            self.progress_signal.emit(f'訓練完成 - LightGBM: {lgb_acc:.4f}, CatBoost: {cb_acc:.4f}, Ensemble: {ensemble_acc:.4f}')
            self.completed_signal.emit(results)
            
        except Exception as e:
            self.error_signal.emit(f'訓練失敗: {str(e)}')


class KlineCanvas(FigureCanvas):
    """K線圖繪製"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_kline(self, df, last_n=1000):
        """繪製K線圖"""
        self.ax.clear()
        
        if len(df) == 0:
            self.ax.text(0.5, 0.5, '無資料', ha='center', va='center')
            self.draw()
            return
        
        # 僅顯示最後 N 根 K 線
        display_df = df.tail(last_n).reset_index(drop=True)
        
        for idx, row in display_df.iterrows():
            x = idx
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # 繪製燭芯
            self.ax.plot([x, x], [low_price, high_price], 'k-', linewidth=0.5)
            
            # 繪製燭身
            body_color = 'green' if close_price >= open_price else 'red'
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            self.ax.add_patch(mpatches.Rectangle(
                (x - 0.3, body_bottom), 0.6, body_height,
                facecolor=body_color, edgecolor='black', linewidth=0.5
            ))
        
        self.ax.set_xlabel('時間')
        self.ax.set_ylabel('價格')
        self.ax.set_title(f'K線圖 (最近 {len(display_df)} 根)')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


class FeatureEngineer:
    """特徵工程"""
    @staticmethod
    def engineer_features(df, lookback=14):
        """生成特徵"""
        df = df.copy()
        
        # 基礎特徵
        df['returns'] = df['close'].pct_change()
        df['high_low'] = (df['high'] - df['low']) / df['close']
        df['close_open'] = (df['close'] - df['open']) / df['open']
        
        # 移動平均
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'rsi_{period}'] = FeatureEngineer._calculate_rsi(df['close'], period)
        
        # 波動性
        df['volatility'] = df['returns'].rolling(window=14).std()
        
        # 標籤: 未來方向
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # 移除 NaN
        df = df.dropna()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """計算 RSI 指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class ModelEnsembleGUI(QMainWindow):
    """加密貨幣模型 Ensemble GUI - HuggingFace 資料源"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('加密貨幣模型 Ensemble GUI - HuggingFace v2-crypto-ohlcv-data')
        self.setGeometry(100, 100, 1400, 900)
        
        self.df_kline = None
        self.df_processed = None
        self.trained_models = None
        self.current_symbol = None
        self.current_timeframe = None
        
        self.loader_worker = None
        self.training_worker = None
        
        self.init_ui()

    def init_ui(self):
        """初始化 UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        # 1. 資料選擇區
        data_group = QGroupBox('資料源 (HuggingFace - zongowo111/v2-crypto-ohlcv-data)')
        data_layout = QGridLayout()
        
        # 幣種選擇
        data_layout.addWidget(QLabel('幣種 (23種):'), 0, 0)
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(SUPPORTED_SYMBOLS)
        self.symbol_combo.setCurrentText('BTCUSDT')
        data_layout.addWidget(self.symbol_combo, 0, 1)
        
        # 時間框選擇
        data_layout.addWidget(QLabel('時間框:'), 0, 2)
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(TIMEFRAMES)
        data_layout.addWidget(self.timeframe_combo, 0, 3)
        
        # 加載按鍵
        self.load_btn = QPushButton('從 HuggingFace 加載')
        self.load_btn.clicked.connect(self.load_hf_data)
        data_layout.addWidget(self.load_btn, 0, 4)
        
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)
        
        # 2. 標籤頁
        self.tabs = QTabWidget()
        
        # 標籤 1: 資料預覽
        self.tab_data = QWidget()
        tab_data_layout = QVBoxLayout()
        
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(['時間', 'Open', 'High', 'Low', 'Close', 'Volume'])
        tab_data_layout.addWidget(self.data_table)
        
        self.tab_data.setLayout(tab_data_layout)
        self.tabs.addTab(self.tab_data, '資料預覽')
        
        # 標籤 2: K 線圖
        self.tab_kline = QWidget()
        tab_kline_layout = QVBoxLayout()
        self.kline_canvas = KlineCanvas(self)
        tab_kline_layout.addWidget(self.kline_canvas)
        self.tab_kline.setLayout(tab_kline_layout)
        self.tabs.addTab(self.tab_kline, 'K線圖')
        
        # 標籤 3: 模型訓練
        self.tab_training = QWidget()
        tab_training_layout = QVBoxLayout()
        
        # 訓練參數
        params_group = QGroupBox('訓練參數')
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel('Lookback 週期:'), 0, 0)
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setValue(14)
        self.lookback_spin.setMinimum(5)
        self.lookback_spin.setMaximum(100)
        params_layout.addWidget(self.lookback_spin, 0, 1)
        
        params_layout.addWidget(QLabel('測試比例:'), 0, 2)
        self.test_ratio_spin = QSpinBox()
        self.test_ratio_spin.setValue(20)
        self.test_ratio_spin.setMinimum(5)
        self.test_ratio_spin.setMaximum(50)
        self.test_ratio_spin.setSuffix(' %')
        params_layout.addWidget(self.test_ratio_spin, 0, 3)
        
        params_group.setLayout(params_layout)
        tab_training_layout.addWidget(params_group)
        
        # 訓練按鍵
        self.train_btn = QPushButton('開始訓練')
        self.train_btn.clicked.connect(self.start_training)
        tab_training_layout.addWidget(self.train_btn)
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        tab_training_layout.addWidget(self.progress_bar)
        
        # 結果顯示
        self.training_text = QTextEdit()
        self.training_text.setReadOnly(True)
        tab_training_layout.addWidget(self.training_text)
        
        # 保存模型
        self.save_model_btn = QPushButton('保存訓練的模型')
        self.save_model_btn.clicked.connect(self.save_models)
        self.save_model_btn.setEnabled(False)
        tab_training_layout.addWidget(self.save_model_btn)
        
        self.tab_training.setLayout(tab_training_layout)
        self.tabs.addTab(self.tab_training, '模型訓練')
        
        # 標籤 4: 預測
        self.tab_prediction = QWidget()
        tab_pred_layout = QVBoxLayout()
        
        self.load_model_btn = QPushButton('加載已訓練的模型')
        self.load_model_btn.clicked.connect(self.load_models)
        tab_pred_layout.addWidget(self.load_model_btn)
        
        self.predict_btn = QPushButton('執行預測')
        self.predict_btn.clicked.connect(self.make_predictions)
        self.predict_btn.setEnabled(False)
        tab_pred_layout.addWidget(self.predict_btn)
        
        self.export_btn = QPushButton('匯出預測結果為 CSV')
        self.export_btn.clicked.connect(self.export_predictions)
        self.export_btn.setEnabled(False)
        tab_pred_layout.addWidget(self.export_btn)
        
        self.pred_text = QTextEdit()
        self.pred_text.setReadOnly(True)
        tab_pred_layout.addWidget(self.pred_text)
        
        self.tab_prediction.setLayout(tab_pred_layout)
        self.tabs.addTab(self.tab_prediction, '預測')
        
        main_layout.addWidget(self.tabs)
        
        # 狀態標籤
        self.status_label = QLabel('就緒 | 資料源: HuggingFace Hub (zongowo111/v2-crypto-ohlcv-data)')
        main_layout.addWidget(self.status_label)
        
        central_widget.setLayout(main_layout)

    def load_hf_data(self):
        """從 HuggingFace 加載資料"""
        symbol = self.symbol_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        
        self.status_label.setText(f'正在加載 {symbol} {timeframe}...')
        self.load_btn.setEnabled(False)
        
        self.loader_worker = DataLoaderWorker(symbol, timeframe)
        self.loader_worker.progress_signal.connect(self.update_status)
        self.loader_worker.completed_signal.connect(self.on_data_loaded)
        self.loader_worker.error_signal.connect(self.on_load_error)
        self.loader_worker.start()

    def update_status(self, message):
        """更新狀態"""
        self.status_label.setText(message)

    def on_data_loaded(self, df):
        """資料加載完成"""
        self.df_kline = df.copy()
        self.current_symbol = self.symbol_combo.currentText()
        self.current_timeframe = self.timeframe_combo.currentText()
        
        # 顯示資料表格
        self.display_data_table(df.head(20))
        
        # 繪製 K 線
        self.kline_canvas.plot_kline(df)
        
        # 特徵工程
        self.df_processed = FeatureEngineer.engineer_features(df, self.lookback_spin.value())
        
        self.status_label.setText(f'已加載 {len(df)} 筆資料，處理後 {len(self.df_processed)} 筆資料')
        self.load_btn.setEnabled(True)
        self.train_btn.setEnabled(True)

    def on_load_error(self, error):
        """加載失敗"""
        QMessageBox.critical(self, '錯誤', error)
        self.load_btn.setEnabled(True)
        self.status_label.setText('加載失敗')

    def display_data_table(self, df):
        """顯示資料表格"""
        self.data_table.setRowCount(len(df))
        
        for row, (idx, record) in enumerate(df.iterrows()):
            self.data_table.setItem(row, 0, QTableWidgetItem(str(record['timestamp'])))
            self.data_table.setItem(row, 1, QTableWidgetItem(f"{record['open']:.2f}"))
            self.data_table.setItem(row, 2, QTableWidgetItem(f"{record['high']:.2f}"))
            self.data_table.setItem(row, 3, QTableWidgetItem(f"{record['low']:.2f}"))
            self.data_table.setItem(row, 4, QTableWidgetItem(f"{record['close']:.2f}"))
            self.data_table.setItem(row, 5, QTableWidgetItem(f"{record['volume']:.0f}"))

    def start_training(self):
        """開始訓練"""
        if self.df_processed is None or len(self.df_processed) < 100:
            QMessageBox.warning(self, '警告', '請先加載足夠的資料')
            return
        
        # 準備資料
        feature_cols = [col for col in self.df_processed.columns 
                       if col not in ['timestamp', 'symbol', 'target']]
        X = self.df_processed[feature_cols].values
        y = self.df_processed['target'].values
        
        # 分割
        test_size = self.test_ratio_spin.value() / 100
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.training_text.clear()
        
        self.training_worker = TrainingWorker(X_train, X_test, y_train, y_test)
        self.training_worker.progress_signal.connect(self.update_training_text)
        self.training_worker.completed_signal.connect(self.on_training_completed)
        self.training_worker.error_signal.connect(self.on_training_error)
        self.training_worker.start()

    def update_training_text(self, message):
        """更新訓練文本"""
        self.training_text.append(f'[{datetime.now().strftime("%H:%M:%S")}] {message}')
        self.progress_bar.setValue(min(100, self.progress_bar.value() + 10))

    def on_training_completed(self, results):
        """訓練完成"""
        self.trained_models = results
        self.train_btn.setEnabled(True)
        self.save_model_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        msg = f"""
訓練完成!

LightGBM 準確度: {results['lgb_acc']:.4f}
CatBoost 準確度: {results['cb_acc']:.4f}
Ensemble 準確度: {results['ensemble_acc']:.4f}

模型已準備好進行預測。
"""
        self.training_text.append(msg)
        self.status_label.setText('訓練完成')

    def on_training_error(self, error):
        """訓練失敗"""
        QMessageBox.critical(self, '錯誤', error)
        self.train_btn.setEnabled(True)
        self.status_label.setText('訓練失敗')

    def save_models(self):
        """保存模型"""
        if self.trained_models is None:
            QMessageBox.warning(self, '警告', '還沒有訓練模型')
            return
        
        save_dir = Path('trained_models')
        save_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol = self.current_symbol.replace('USDT', '')
        timeframe = self.current_timeframe
        
        # 保存模型
        lgb_path = save_dir / f'lightgbm_{symbol}_{timeframe}_{timestamp}.pkl'
        cb_path = save_dir / f'catboost_{symbol}_{timeframe}_{timestamp}.pkl'
        scaler_path = save_dir / f'scaler_{symbol}_{timeframe}_{timestamp}.pkl'
        
        with open(lgb_path, 'wb') as f:
            pickle.dump(self.trained_models['lgb_model'], f)
        
        with open(cb_path, 'wb') as f:
            pickle.dump(self.trained_models['cb_model'], f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.trained_models['scaler'], f)
        
        QMessageBox.information(self, '成功', f'模型已保存到 {save_dir}')
        self.status_label.setText(f'模型已保存')

    def load_models(self):
        """加載模型"""
        file_dialog = QFileDialog()
        file_dialog.setDirectory('trained_models')
        
        lgb_file, _ = file_dialog.getOpenFileName(self, '選擇 LightGBM 模型', 'trained_models', '*.pkl')
        if not lgb_file:
            return
        
        try:
            with open(lgb_file, 'rb') as f:
                lgb_model = pickle.load(f)
            
            # 自動尋找對應的 CatBoost 和 Scaler
            base_path = Path(lgb_file).parent
            base_name = Path(lgb_file).stem.replace('lightgbm_', '')
            
            cb_files = list(base_path.glob(f'catboost_{base_name}*.pkl'))
            scaler_files = list(base_path.glob(f'scaler_{base_name}*.pkl'))
            
            if not cb_files or not scaler_files:
                QMessageBox.warning(self, '警告', '找不到對應的 CatBoost 或 Scaler 文件')
                return
            
            with open(cb_files[0], 'rb') as f:
                cb_model = pickle.load(f)
            
            with open(scaler_files[0], 'rb') as f:
                scaler = pickle.load(f)
            
            self.trained_models = {
                'lgb_model': lgb_model,
                'cb_model': cb_model,
                'scaler': scaler
            }
            
            QMessageBox.information(self, '成功', '模型加載成功')
            self.predict_btn.setEnabled(True)
            self.export_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, '錯誤', f'加載失敗: {str(e)}')

    def make_predictions(self):
        """執行預測"""
        if self.trained_models is None or self.df_processed is None:
            QMessageBox.warning(self, '警告', '請先訓練或加載模型，並加載資料')
            return
        
        try:
            feature_cols = [col for col in self.df_processed.columns 
                           if col not in ['timestamp', 'symbol', 'target']]
            X = self.df_processed[feature_cols].values
            
            scaler = self.trained_models['scaler']
            X_scaled = scaler.transform(X)
            
            lgb_model = self.trained_models['lgb_model']
            cb_model = self.trained_models['cb_model']
            
            lgb_pred = lgb_model.predict(X_scaled)
            lgb_proba = lgb_model.predict_proba(X_scaled)[:, 1]
            
            cb_pred = cb_model.predict(X_scaled)
            cb_proba = cb_model.predict_proba(X_scaled)[:, 1]
            
            ensemble_pred = ((lgb_proba + cb_proba) / 2 > 0.5).astype(int)
            ensemble_proba = (lgb_proba + cb_proba) / 2
            
            # 保存預測結果
            self.prediction_results = pd.DataFrame({
                'timestamp': self.df_processed['timestamp'],
                'lgb_prediction': lgb_pred,
                'lgb_probability': lgb_proba,
                'cb_prediction': cb_pred,
                'cb_probability': cb_proba,
                'ensemble_prediction': ensemble_pred,
                'ensemble_probability': ensemble_proba
            })
            
            # 顯示統計
            msg = f"""
預測完成!

總筆數: {len(self.prediction_results)}
上升預測 (Ensemble): {(ensemble_pred == 1).sum()} ({(ensemble_pred == 1).sum()/len(ensemble_pred)*100:.2f}%)
下降預測 (Ensemble): {(ensemble_pred == 0).sum()} ({(ensemble_pred == 0).sum()/len(ensemble_pred)*100:.2f}%)

平均概率 (Ensemble): {ensemble_proba.mean():.4f}
最後 10 筆預測:
{self.prediction_results.tail(10).to_string()}
"""
            self.pred_text.setText(msg)
            self.export_btn.setEnabled(True)
            self.status_label.setText('預測完成')
            
        except Exception as e:
            QMessageBox.critical(self, '錯誤', f'預測失敗: {str(e)}')

    def export_predictions(self):
        """匯出預測"""
        if self.prediction_results is None:
            QMessageBox.warning(self, '警告', '還沒有預測結果')
            return
        
        file_dialog = QFileDialog()
        csv_file, _ = file_dialog.getSaveFileName(self, '保存預測結果', '', '*.csv')
        
        if csv_file:
            self.prediction_results.to_csv(csv_file, index=False)
            QMessageBox.information(self, '成功', f'預測結果已保存到 {csv_file}')


def main():
    app = QApplication(sys.argv)
    gui = ModelEnsembleGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
