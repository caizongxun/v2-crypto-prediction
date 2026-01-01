import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QFrame, QProgressBar, QMessageBox, QTabWidget, QSpinBox,
    QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
import logging
from indicators import IndicatorCalculator
from chart_renderer import ChartRenderer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HuggingFace 數據集配置
HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
HF_REPO_TYPE = "dataset"

# 支持的幣種列表（來自圖片中的列表）
SUPPORTED_SYMBOLS = [
    "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT",
    "AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "DOTUSDT", "ETCUSDT", "ETHUSDT", "FILUSDT", "LINKUSDT",
    "LTCUSDT", "MATICUSDT", "NEARUSDT", "OPUSDT", "SOLUSDT",
    "UNIUSDT", "XRPUSDT"
]

# K 線時間框架
TIMEFRAMES = ["15m", "1h"]


class KlineDataFetcher:
    """從 HuggingFace 數據集獲取 K 線數據"""
    
    def __init__(self):
        self.cache = {}
        self.last_fetch_time = {}
    
    def fetch_kline_data(self, symbol, timeframe="15m", use_cache=True):
        """
        從 HuggingFace 數據集下載並讀取 K 線數據
        
        Args:
            symbol (str): 幣種符號，例如 "BTCUSDT"
            timeframe (str): 時間框架，"15m" 或 "1h"
            use_cache (bool): 是否使用緩存
            
        Returns:
            pd.DataFrame: 包含 OHLCV 數據的 DataFrame
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # 檢查緩存
        if use_cache and cache_key in self.cache:
            logger.info(f"使用緩存數據: {cache_key}")
            return self.cache[cache_key]
        
        try:
            # 構造文件名
            # 例如: BTCUSDT -> klines/BTCUSDT/BTC_15m.parquet
            symbol_prefix = symbol.replace("USDT", "")
            filename = f"klines/{symbol}/{symbol_prefix}_{timeframe}.parquet"
            
            logger.info(f"正在從 HuggingFace 下載: {filename}")
            
            # 從 HuggingFace 下載文件
            file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type=HF_REPO_TYPE,
                force_download=False  # 使用緩存
            )
            
            # 讀取 Parquet 文件
            df = pd.read_parquet(file_path)
            
            # 標準化列名和數據格式
            df = self._standardize_dataframe(df)
            
            # 存儲到緩存
            self.cache[cache_key] = df
            self.last_fetch_time[cache_key] = datetime.now()
            
            logger.info(f"成功加載 {symbol} {timeframe} 數據: {len(df)} 條記錄")
            return df
            
        except Exception as e:
            logger.error(f"下載 {symbol} {timeframe} 數據失敗: {str(e)}")
            raise
    
    def _standardize_dataframe(self, df):
        """標準化 DataFrame 格式"""
        # 確保列名為小寫
        df.columns = df.columns.str.lower()
        
        # 重命名常見的列名
        rename_map = {
            'open_time': 'time',
            'timestamp': 'time',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'volume': 'volume'
        }
        
        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # 確保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"缺少列: {col}")
        
        # 轉換數據類型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 處理時間列
        if 'time' in df.columns:
            if df['time'].dtype == 'object':
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            elif df['time'].dtype in ['int64', 'float64']:
                # 假設是毫秒時間戳
                df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        
        # 按時間排序
        if 'time' in df.columns:
            df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def get_latest_kline(self, symbol, timeframe="15m"):
        """獲取最新的 K 線"""
        df = self.fetch_kline_data(symbol, timeframe)
        if df is not None and len(df) > 0:
            return df.iloc[-1].to_dict()
        return None
    
    def get_kline_range(self, symbol, timeframe="15m", start_date=None, end_date=None):
        """獲取指定日期範圍的 K 線數據"""
        df = self.fetch_kline_data(symbol, timeframe)
        if df is None or len(df) == 0:
            return None
        
        if 'time' in df.columns:
            if start_date:
                df = df[df['time'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['time'] <= pd.to_datetime(end_date)]
        
        return df
    
    def clear_cache(self, symbol=None, timeframe=None):
        """清空緩存"""
        if symbol is None:
            self.cache.clear()
            logger.info("已清空所有緩存")
        else:
            cache_key = f"{symbol}_{timeframe}" if timeframe else symbol
            if cache_key in self.cache:
                del self.cache[cache_key]
                logger.info(f"已清空緩存: {cache_key}")


class DataLoadThread(QThread):
    """后台數據加載線程"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    data_loaded = pyqtSignal(pd.DataFrame, dict)
    
    def __init__(self, fetcher, symbol, timeframe):
        super().__init__()
        self.fetcher = fetcher
        self.symbol = symbol
        self.timeframe = timeframe
    
    def run(self):
        try:
            df = self.fetcher.fetch_kline_data(self.symbol, self.timeframe)
            
            if df is None or len(df) == 0:
                self.error.emit("無法加載數據或數據為空")
                return
            
            # 計算統計信息
            stats = {
                'total_records': len(df),
                'start_time': df['time'].min() if 'time' in df.columns else None,
                'end_time': df['time'].max() if 'time' in df.columns else None,
                'current_price': df['close'].iloc[-1] if 'close' in df.columns else None,
                'high_24h': df['high'].max() if 'high' in df.columns else None,
                'low_24h': df['low'].min() if 'low' in df.columns else None,
            }
            
            self.data_loaded.emit(df, stats)
            self.finished.emit()
            
        except Exception as e:
            logger.error(f"加載數據出錯: {str(e)}")
            self.error.emit(f"加載數據出錯: {str(e)}")


class ChartRenderThread(QThread):
    """后台圖表繪製線程"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    chart_ready = pyqtSignal(Figure)
    
    def __init__(self, df: pd.DataFrame, symbol: str, num_candles: int = 300,
                 show_fib: bool = True, show_ob: bool = True):
        super().__init__()
        self.df = df
        self.symbol = symbol
        self.num_candles = num_candles
        self.show_fib = show_fib
        self.show_ob = show_ob
    
    def run(self):
        try:
            calculator = IndicatorCalculator()
            
            # 識別訂單塊
            order_blocks = None
            if self.show_ob:
                order_blocks = calculator.identify_order_blocks(
                    self.df, periods=5, threshold=0.0
                )
            
            # 識別振盪高低點用於斐波那契
            fib_levels = None
            if self.show_fib:
                swing_highs, swing_lows = calculator.identify_swing_highs_lows(
                    self.df, period=3
                )
                
                # 使用最近的高低點計算斐波那契
                if swing_highs and swing_lows:
                    last_high_idx = swing_highs[-1]\n                    last_low_idx = swing_lows[-1]\n                    \n                    high_price = self.df['high'].iloc[last_high_idx]\n                    low_price = self.df['low'].iloc[last_low_idx]\n                    \n                    if last_high_idx > last_low_idx:\n                        # 看跌趨勢\n                        fib_levels = calculator.calculate_fibonacci_levels(\n                            last_high_idx, last_low_idx, high_price, low_price, \n                            is_bullish=False\n                        )\n                    else:\n                        # 看漲趨勢\n                        fib_levels = calculator.calculate_fibonacci_levels(\n                            last_high_idx, last_low_idx, high_price, low_price, \n                            is_bullish=True\n                        )\n            \n            # 創建圖表\n            fig = ChartRenderer.create_candlestick_chart(\n                self.df,\n                num_candles=self.num_candles,\n                order_blocks=order_blocks,\n                fib_levels=fib_levels,\n                title=f'{self.symbol} - K線圖表 (最後 {self.num_candles} 根)'\n            )\n            \n            self.chart_ready.emit(fig)\n            self.finished.emit()\n            \n        except Exception as e:\n            logger.error(f\"繪製圖表出錯: {str(e)}\")\n            self.error.emit(f\"繪製圖表出錯: {str(e)}\")\n\n\nclass ModelEnsembleGUI(QMainWindow):\n    \"\"\"模型集成 GUI 界面\"\"\"\n    \n    def __init__(self):\n        super().__init__()\n        self.setWindowTitle(\"加密貨幣 K 線數據查詢系統\")\n        self.setGeometry(100, 100, 1400, 900)\n        \n        self.fetcher = KlineDataFetcher()\n        self.current_data = None\n        self.load_thread = None\n        self.chart_thread = None\n        \n        self.init_ui()\n    \n    def init_ui(self):\n        \"\"\"初始化 UI\"\"\"\n        central_widget = QWidget()\n        self.setCentralWidget(central_widget)\n        \n        main_layout = QVBoxLayout(central_widget)\n        main_layout.setContentsMargins(10, 10, 10, 10)\n        main_layout.setSpacing(10)\n        \n        # 控制面板\n        control_layout = QHBoxLayout()\n        \n        # 幣種選擇\n        control_layout.addWidget(QLabel(\"選擇幣種:\"))\n        self.symbol_combo = QComboBox()\n        self.symbol_combo.addItems(SUPPORTED_SYMBOLS)\n        self.symbol_combo.setCurrentText(\"BTCUSDT\")\n        control_layout.addWidget(self.symbol_combo)\n        \n        # 時間框架選擇\n        control_layout.addWidget(QLabel(\"時間框架:\"))\n        self.timeframe_combo = QComboBox()\n        self.timeframe_combo.addItems(TIMEFRAMES)\n        self.timeframe_combo.setCurrentText(\"15m\")\n        control_layout.addWidget(self.timeframe_combo)\n        \n        # K線數量選擇\n        control_layout.addWidget(QLabel(\"K線數量:\"))\n        self.candle_spinbox = QSpinBox()\n        self.candle_spinbox.setMinimum(50)\n        self.candle_spinbox.setMaximum(300)\n        self.candle_spinbox.setValue(300)\n        self.candle_spinbox.setSingleStep(50)\n        control_layout.addWidget(self.candle_spinbox)\n        \n        # 指標選擇\n        self.fib_checkbox = QCheckBox(\"斐波那契\")\n        self.fib_checkbox.setChecked(True)\n        control_layout.addWidget(self.fib_checkbox)\n        \n        self.ob_checkbox = QCheckBox(\"訂單塊\")\n        self.ob_checkbox.setChecked(True)\n        control_layout.addWidget(self.ob_checkbox)\n        \n        # 加載按鈕\n        self.load_btn = QPushButton(\"加載數據\")\n        self.load_btn.clicked.connect(self.load_kline_data)\n        control_layout.addWidget(self.load_btn)\n        \n        # 繪製圖表按鈕\n        self.chart_btn = QPushButton(\"繪製圖表\")\n        self.chart_btn.clicked.connect(self.render_chart)\n        self.chart_btn.setEnabled(False)\n        control_layout.addWidget(self.chart_btn)\n        \n        # 清空緩存按鈕\n        self.clear_cache_btn = QPushButton(\"清空緩存\")\n        self.clear_cache_btn.clicked.connect(self.clear_cache)\n        control_layout.addWidget(self.clear_cache_btn)\n        \n        control_layout.addStretch()\n        main_layout.addLayout(control_layout)\n        \n        # 進度條\n        self.progress_bar = QProgressBar()\n        self.progress_bar.setMaximum(0)\n        self.progress_bar.setVisible(False)\n        main_layout.addWidget(self.progress_bar)\n        \n        # Tab 小部件\n        self.tabs = QTabWidget()\n        \n        # Tab 1: 數據表格\n        self.table_widget = self.create_data_tab()\n        self.tabs.addTab(self.table_widget, \"K線數據\")\n        \n        # Tab 2: 圖表\n        self.chart_widget = self.create_chart_tab()\n        self.tabs.addTab(self.chart_widget, \"K線圖表\")\n        \n        main_layout.addWidget(self.tabs)\n        \n        central_widget.setLayout(main_layout)\n    \n    def create_data_tab(self):\n        \"\"\"創建數據表格選項卡\"\"\"\n        tab_widget = QWidget()\n        layout = QVBoxLayout(tab_widget)\n        layout.setContentsMargins(0, 0, 0, 0)\n        \n        # 信息面板\n        info_frame = QFrame()\n        info_layout = QVBoxLayout(info_frame)\n        info_layout.setContentsMargins(0, 0, 0, 0)\n        \n        info_label = QLabel(\"數據信息\")\n        font = QFont()\n        font.setBold(True)\n        info_label.setFont(font)\n        info_layout.addWidget(info_label)\n        \n        self.info_text = QTextEdit()\n        self.info_text.setReadOnly(True)\n        self.info_text.setMaximumHeight(120)\n        info_layout.addWidget(self.info_text)\n        \n        layout.addWidget(info_frame)\n        \n        # 數據表格\n        table_label = QLabel(\"K 線數據\")\n        table_label.setFont(font)\n        layout.addWidget(table_label)\n        \n        self.table = QTableWidget()\n        self.table.setColumnCount(6)\n        self.table.setHorizontalHeaderLabels([\"時間\", \"開盤\", \"最高\", \"最低\", \"收盤\", \"成交量\"])\n        self.table.horizontalHeader().setStretchLastSection(True)\n        layout.addWidget(self.table)\n        \n        return tab_widget\n    \n    def create_chart_tab(self):\n        \"\"\"創建圖表選項卡\"\"\"\n        tab_widget = QWidget()\n        layout = QVBoxLayout(tab_widget)\n        layout.setContentsMargins(0, 0, 0, 0)\n        \n        # 圖表容器\n        self.chart_canvas_widget = QWidget()\n        self.chart_layout = QVBoxLayout(self.chart_canvas_widget)\n        self.chart_layout.setContentsMargins(0, 0, 0, 0)\n        \n        layout.addWidget(self.chart_canvas_widget)\n        \n        return tab_widget\n    \n    def load_kline_data(self):\n        \"\"\"加載 K 線數據\"\"\"\n        if self.load_thread is not None and self.load_thread.isRunning():\n            QMessageBox.warning(self, \"警告\", \"數據加載中，請稍候\")\n            return\n        \n        symbol = self.symbol_combo.currentText()\n        timeframe = self.timeframe_combo.currentText()\n        \n        # 顯示進度條\n        self.progress_bar.setVisible(True)\n        self.load_btn.setEnabled(False)\n        self.chart_btn.setEnabled(False)\n        self.clear_cache_btn.setEnabled(False)\n        self.info_text.setText(f\"正在加載 {symbol} {timeframe} 數據...\\n\")\n        \n        # 創建后台線程\n        self.load_thread = DataLoadThread(self.fetcher, symbol, timeframe)\n        self.load_thread.data_loaded.connect(self.on_data_loaded)\n        self.load_thread.error.connect(self.on_error)\n        self.load_thread.finished.connect(self.on_finished)\n        self.load_thread.start()\n    \n    def render_chart(self):\n        \"\"\"繪製圖表\"\"\"\n        if self.current_data is None:\n            QMessageBox.warning(self, \"警告\", \"請先加載數據\")\n            return\n        \n        if self.chart_thread is not None and self.chart_thread.isRunning():\n            QMessageBox.warning(self, \"警告\", \"圖表繪製中，請稍候\")\n            return\n        \n        symbol = self.symbol_combo.currentText()\n        num_candles = self.candle_spinbox.value()\n        show_fib = self.fib_checkbox.isChecked()\n        show_ob = self.ob_checkbox.isChecked()\n        \n        # 顯示進度條\n        self.progress_bar.setVisible(True)\n        self.chart_btn.setEnabled(False)\n        \n        # 創建繪製線程\n        self.chart_thread = ChartRenderThread(\n            self.current_data, symbol, num_candles, show_fib, show_ob\n        )\n        self.chart_thread.chart_ready.connect(self.on_chart_ready)\n        self.chart_thread.error.connect(self.on_error)\n        self.chart_thread.finished.connect(self.on_chart_finished)\n        self.chart_thread.start()\n    \n    def on_data_loaded(self, df, stats):\n        \"\"\"數據加載完成回調\"\"\"\n        self.current_data = df\n        \n        # 更新信息面板\n        info_text = f\"\"\"幣種: {self.symbol_combo.currentText()}\n時間框架: {self.timeframe_combo.currentText()}\n總記錄數: {stats['total_records']}\n時間範圍: {stats['start_time']} 至 {stats['end_time']}\n當前價格: ${stats['current_price']:.2f}\n24H 高: ${stats['high_24h']:.2f}\n24H 低: ${stats['low_24h']:.2f}\"\"\"\n        \n        self.info_text.setText(info_text)\n        \n        # 更新表格\n        self.update_table(df)\n        \n        # 啟用圖表按鈕\n        self.chart_btn.setEnabled(True)\n    \n    def on_error(self, error_msg):\n        \"\"\"加載出錯回調\"\"\"\n        self.info_text.setText(f\"錯誤: {error_msg}\")\n        QMessageBox.critical(self, \"錯誤\", error_msg)\n    \n    def on_finished(self):\n        \"\"\"加載完成回調\"\"\"\n        self.progress_bar.setVisible(False)\n        self.load_btn.setEnabled(True)\n        self.clear_cache_btn.setEnabled(True)\n    \n    def on_chart_ready(self, fig):\n        \"\"\"圖表準備完成回調\"\"\"\n        # 清空舊圖表\n        for i in reversed(range(self.chart_layout.count())):\n            self.chart_layout.itemAt(i).widget().setParent(None)\n        \n        # 添加新圖表\n        canvas = FigureCanvas(fig)\n        self.chart_layout.addWidget(canvas)\n        canvas.draw()\n    \n    def on_chart_finished(self):\n        \"\"\"圖表繪製完成回調\"\"\"\n        self.progress_bar.setVisible(False)\n        self.chart_btn.setEnabled(True)\n    \n    def update_table(self, df):\n        \"\"\"更新表格\"\"\"\n        self.table.setRowCount(0)\n        \n        # 只顯示最后 100 條記錄\n        df_display = df.tail(100)\n        \n        for idx, row in df_display.iterrows():\n            row_position = self.table.rowCount()\n            self.table.insertRow(row_position)\n            \n            # 時間\n            time_str = row['time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in row and pd.notna(row['time']) else \"N/A\"\n            self.table.setItem(row_position, 0, QTableWidgetItem(time_str))\n            \n            # 開盤\n            open_price = f\"{row['open']:.2f}\" if 'open' in row and pd.notna(row['open']) else \"N/A\"\n            self.table.setItem(row_position, 1, QTableWidgetItem(open_price))\n            \n            # 最高\n            high_price = f\"{row['high']:.2f}\" if 'high' in row and pd.notna(row['high']) else \"N/A\"\n            self.table.setItem(row_position, 2, QTableWidgetItem(high_price))\n            \n            # 最低\n            low_price = f\"{row['low']:.2f}\" if 'low' in row and pd.notna(row['low']) else \"N/A\"\n            self.table.setItem(row_position, 3, QTableWidgetItem(low_price))\n            \n            # 收盤\n            close_price = f\"{row['close']:.2f}\" if 'close' in row and pd.notna(row['close']) else \"N/A\"\n            self.table.setItem(row_position, 4, QTableWidgetItem(close_price))\n            \n            # 成交量\n            volume = f\"{row['volume']:.0f}\" if 'volume' in row and pd.notna(row['volume']) else \"N/A\"\n            self.table.setItem(row_position, 5, QTableWidgetItem(volume))\n    \n    def clear_cache(self):\n        \"\"\"清空緩存\"\"\"\n        self.fetcher.clear_cache()\n        self.info_text.setText(\"已清空所有緩存\")\n        QMessageBox.information(self, \"成功\", \"已清空所有緩存\")\n\n\ndef main():\n    app = QApplication(sys.argv)\n    window = ModelEnsembleGUI()\n    window.show()\n    sys.exit(app.exec_())\n\n\nif __name__ == \"__main__\":\n    main()\n