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
from PyQt5.QtGui import QFont
import logging
from indicators import IndicatorCalculator
from chart_renderer import ChartRenderer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
HF_REPO_TYPE = "dataset"

SUPPORTED_SYMBOLS = [
    "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT",
    "AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "DOTUSDT", "ETCUSDT", "ETHUSDT", "FILUSDT", "LINKUSDT",
    "LTCUSDT", "MATICUSDT", "NEARUSDT", "OPUSDT", "SOLUSDT",
    "UNIUSDT", "XRPUSDT"
]

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
        
        if use_cache and cache_key in self.cache:
            logger.info(f"使用緩存數據: {cache_key}")
            return self.cache[cache_key]
        
        try:
            symbol_prefix = symbol.replace("USDT", "")
            filename = f"klines/{symbol}/{symbol_prefix}_{timeframe}.parquet"
            
            logger.info(f"正在從 HuggingFace 下載: {filename}")
            
            file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type=HF_REPO_TYPE,
                force_download=False
            )
            
            df = pd.read_parquet(file_path)
            df = self._standardize_dataframe(df)
            
            self.cache[cache_key] = df
            self.last_fetch_time[cache_key] = datetime.now()
            
            logger.info(f"成功加載 {symbol} {timeframe} 數據: {len(df)} 條記錄")
            return df
            
        except Exception as e:
            logger.error(f"下載 {symbol} {timeframe} 數據失敗: {str(e)}")
            raise
    
    def _standardize_dataframe(self, df):
        """標準化 DataFrame 格式"""
        df.columns = df.columns.str.lower()
        
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
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"缺少列: {col}")
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'time' in df.columns:
            if df['time'].dtype == 'object':
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            elif df['time'].dtype in ['int64', 'float64']:
                df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        
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
    """后台圖表渲染線程"""
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
            
            order_blocks = None
            if self.show_ob:
                order_blocks = calculator.identify_order_blocks(
                    self.df, periods=5, threshold=0.0
                )
            
            fib_levels = None
            if self.show_fib:
                swing_highs, swing_lows = calculator.identify_swing_highs_lows(
                    self.df, period=3
                )
                
                if swing_highs and swing_lows:
                    last_high_idx = swing_highs[-1]
                    last_low_idx = swing_lows[-1]
                    
                    high_price = self.df['high'].iloc[last_high_idx]
                    low_price = self.df['low'].iloc[last_low_idx]
                    
                    if last_high_idx > last_low_idx:
                        fib_levels = calculator.calculate_fibonacci_levels(
                            last_high_idx, last_low_idx, high_price, low_price, 
                            is_bullish=False
                        )
                    else:
                        fib_levels = calculator.calculate_fibonacci_levels(
                            last_high_idx, last_low_idx, high_price, low_price, 
                            is_bullish=True
                        )
            
            fig = ChartRenderer.create_candlestick_chart(
                self.df,
                num_candles=self.num_candles,
                order_blocks=order_blocks,
                fib_levels=fib_levels,
                title=f'{self.symbol} - K線圖表 (最後 {self.num_candles} 根)'
            )
            
            self.chart_ready.emit(fig)
            self.finished.emit()
            
        except Exception as e:
            logger.error(f"繪製圖表出錯: {str(e)}")
            self.error.emit(f"繪製圖表出錯: {str(e)}")


class ModelEnsembleGUI(QMainWindow):
    """模型集成 GUI 界面"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("加密貨幣 K 線數據查詢系統")
        self.setGeometry(100, 100, 1400, 900)
        
        self.fetcher = KlineDataFetcher()
        self.current_data = None
        self.load_thread = None
        self.chart_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        """初始化 UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("選擇幣種:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(SUPPORTED_SYMBOLS)
        self.symbol_combo.setCurrentText("BTCUSDT")
        control_layout.addWidget(self.symbol_combo)
        
        control_layout.addWidget(QLabel("時間框架:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(TIMEFRAMES)
        self.timeframe_combo.setCurrentText("15m")
        control_layout.addWidget(self.timeframe_combo)
        
        control_layout.addWidget(QLabel("K線數量:"))
        self.candle_spinbox = QSpinBox()
        self.candle_spinbox.setMinimum(50)
        self.candle_spinbox.setMaximum(300)
        self.candle_spinbox.setValue(300)
        self.candle_spinbox.setSingleStep(50)
        control_layout.addWidget(self.candle_spinbox)
        
        self.fib_checkbox = QCheckBox("斐波那契")
        self.fib_checkbox.setChecked(True)
        control_layout.addWidget(self.fib_checkbox)
        
        self.ob_checkbox = QCheckBox("訂單塊")
        self.ob_checkbox.setChecked(True)
        control_layout.addWidget(self.ob_checkbox)
        
        self.load_btn = QPushButton("加載數據")
        self.load_btn.clicked.connect(self.load_kline_data)
        control_layout.addWidget(self.load_btn)
        
        self.chart_btn = QPushButton("繪製圖表")
        self.chart_btn.clicked.connect(self.render_chart)
        self.chart_btn.setEnabled(False)
        control_layout.addWidget(self.chart_btn)
        
        self.clear_cache_btn = QPushButton("清空緩存")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        control_layout.addWidget(self.clear_cache_btn)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.tabs = QTabWidget()
        
        self.table_widget = self.create_data_tab()
        self.tabs.addTab(self.table_widget, "K線數據")
        
        self.chart_widget = self.create_chart_tab()
        self.tabs.addTab(self.chart_widget, "K線圖表")
        
        main_layout.addWidget(self.tabs)
        
        central_widget.setLayout(main_layout)
    
    def create_data_tab(self):
        """創建數據表格選項卡"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        info_frame = QFrame()
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        info_label = QLabel("數據信息")
        font = QFont()
        font.setBold(True)
        info_label.setFont(font)
        info_layout.addWidget(info_label)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_frame)
        
        table_label = QLabel("K 線數據")
        table_label.setFont(font)
        layout.addWidget(table_label)
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["時間", "開盤", "最高", "最低", "收盤", "成交量"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        
        return tab_widget
    
    def create_chart_tab(self):
        """創建圖表選項卡"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.chart_canvas_widget = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_canvas_widget)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(self.chart_canvas_widget)
        
        return tab_widget
    
    def load_kline_data(self):
        """加載 K 線數據"""
        if self.load_thread is not None and self.load_thread.isRunning():
            QMessageBox.warning(self, "警告", "數據加載中，請稍候")
            return
        
        symbol = self.symbol_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        
        self.progress_bar.setVisible(True)
        self.load_btn.setEnabled(False)
        self.chart_btn.setEnabled(False)
        self.clear_cache_btn.setEnabled(False)
        self.info_text.setText(f"正在加載 {symbol} {timeframe} 數據...\n")
        
        self.load_thread = DataLoadThread(self.fetcher, symbol, timeframe)
        self.load_thread.data_loaded.connect(self.on_data_loaded)
        self.load_thread.error.connect(self.on_error)
        self.load_thread.finished.connect(self.on_finished)
        self.load_thread.start()
    
    def render_chart(self):
        """繪製圖表"""
        if self.current_data is None:
            QMessageBox.warning(self, "警告", "請先加載數據")
            return
        
        if self.chart_thread is not None and self.chart_thread.isRunning():
            QMessageBox.warning(self, "警告", "圖表繪製中，請稍候")
            return
        
        symbol = self.symbol_combo.currentText()
        num_candles = self.candle_spinbox.value()
        show_fib = self.fib_checkbox.isChecked()
        show_ob = self.ob_checkbox.isChecked()
        
        self.progress_bar.setVisible(True)
        self.chart_btn.setEnabled(False)
        
        self.chart_thread = ChartRenderThread(
            self.current_data, symbol, num_candles, show_fib, show_ob
        )
        self.chart_thread.chart_ready.connect(self.on_chart_ready)
        self.chart_thread.error.connect(self.on_error)
        self.chart_thread.finished.connect(self.on_chart_finished)
        self.chart_thread.start()
    
    def on_data_loaded(self, df, stats):
        """數據加載完成回調"""
        self.current_data = df
        
        info_text = f"""幣種: {self.symbol_combo.currentText()}
時間框架: {self.timeframe_combo.currentText()}
總記錄數: {stats['total_records']}
時間範圍: {stats['start_time']} 至 {stats['end_time']}
當前價格: ${stats['current_price']:.2f}
24H 高: ${stats['high_24h']:.2f}
24H 低: ${stats['low_24h']:.2f}"""
        
        self.info_text.setText(info_text)
        self.update_table(df)
        self.chart_btn.setEnabled(True)
    
    def on_error(self, error_msg):
        """加載出錯回調"""
        self.info_text.setText(f"錯誤: {error_msg}")
        QMessageBox.critical(self, "錯誤", error_msg)
    
    def on_finished(self):
        """加載完成回調"""
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        self.clear_cache_btn.setEnabled(True)
    
    def on_chart_ready(self, fig):
        """圖表準備完成回調"""
        for i in reversed(range(self.chart_layout.count())):
            self.chart_layout.itemAt(i).widget().setParent(None)
        
        canvas = FigureCanvas(fig)
        self.chart_layout.addWidget(canvas)
        canvas.draw()
    
    def on_chart_finished(self):
        """圖表繪製完成回調"""
        self.progress_bar.setVisible(False)
        self.chart_btn.setEnabled(True)
    
    def update_table(self, df):
        """更新表格"""
        self.table.setRowCount(0)
        
        df_display = df.tail(100)
        
        for idx, row in df_display.iterrows():
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            
            time_str = row['time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in row and pd.notna(row['time']) else "N/A"
            self.table.setItem(row_position, 0, QTableWidgetItem(time_str))
            
            open_price = f"{row['open']:.2f}" if 'open' in row and pd.notna(row['open']) else "N/A"
            self.table.setItem(row_position, 1, QTableWidgetItem(open_price))
            
            high_price = f"{row['high']:.2f}" if 'high' in row and pd.notna(row['high']) else "N/A"
            self.table.setItem(row_position, 2, QTableWidgetItem(high_price))
            
            low_price = f"{row['low']:.2f}" if 'low' in row and pd.notna(row['low']) else "N/A"
            self.table.setItem(row_position, 3, QTableWidgetItem(low_price))
            
            close_price = f"{row['close']:.2f}" if 'close' in row and pd.notna(row['close']) else "N/A"
            self.table.setItem(row_position, 4, QTableWidgetItem(close_price))
            
            volume = f"{row['volume']:.0f}" if 'volume' in row and pd.notna(row['volume']) else "N/A"
            self.table.setItem(row_position, 5, QTableWidgetItem(volume))
    
    def clear_cache(self):
        """清空緩存"""
        self.fetcher.clear_cache()
        self.info_text.setText("已清空所有緩存")
        QMessageBox.information(self, "成功", "已清空所有緩存")


def main():
    app = QApplication(sys.argv)
    window = ModelEnsembleGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
