import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
import logging

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

# 嘗試導入 PyQt5，如果失敗則使用 Tkinter
use_pyqt5 = True
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
        QTextEdit, QFrame, QProgressBar, QMessageBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    logger.info("成功加載 PyQt5")
except (ImportError, OSError) as e:
    logger.warning(f"無法加載 PyQt5: {str(e)}")
    logger.info("將使用 Tkinter 替代方案")
    use_pyqt5 = False


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


if use_pyqt5:
    # ==================== PyQt5 版本 ====================
    
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
    
    
    class ModelEnsembleGUI(QMainWindow):
        """模型集成 GUI 界面"""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("加密貨幣 K 線數據查詢系統")
            self.setGeometry(100, 100, 1200, 800)
            
            self.fetcher = KlineDataFetcher()
            self.current_data = None
            self.load_thread = None
            
            self.init_ui()
        
        def init_ui(self):
            """初始化 UI"""
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            main_layout = QVBoxLayout(central_widget)
            main_layout.setContentsMargins(10, 10, 10, 10)
            main_layout.setSpacing(10)
            
            # 控制面板
            control_layout = QHBoxLayout()
            
            # 幣種選擇
            control_layout.addWidget(QLabel("選擇幣種:"))
            self.symbol_combo = QComboBox()
            self.symbol_combo.addItems(SUPPORTED_SYMBOLS)
            self.symbol_combo.setCurrentText("BTCUSDT")
            control_layout.addWidget(self.symbol_combo)
            
            # 時間框架選擇
            control_layout.addWidget(QLabel("時間框架:"))
            self.timeframe_combo = QComboBox()
            self.timeframe_combo.addItems(TIMEFRAMES)
            self.timeframe_combo.setCurrentText("15m")
            control_layout.addWidget(self.timeframe_combo)
            
            # 加載按鈕
            self.load_btn = QPushButton("加載數據")
            self.load_btn.clicked.connect(self.load_kline_data)
            control_layout.addWidget(self.load_btn)
            
            # 清空緩存按鈕
            self.clear_cache_btn = QPushButton("清空緩存")
            self.clear_cache_btn.clicked.connect(self.clear_cache)
            control_layout.addWidget(self.clear_cache_btn)
            
            control_layout.addStretch()
            main_layout.addLayout(control_layout)
            
            # 進度條
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximum(0)
            self.progress_bar.setVisible(False)
            main_layout.addWidget(self.progress_bar)
            
            # 信息面板
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
            
            main_layout.addWidget(info_frame)
            
            # 數據表格
            table_label = QLabel("K 線數據")
            table_label.setFont(font)
            main_layout.addWidget(table_label)
            
            self.table = QTableWidget()
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(["時間", "開盤", "最高", "最低", "收盤", "成交量"])
            self.table.horizontalHeader().setStretchLastSection(True)
            main_layout.addWidget(self.table)
            
            central_widget.setLayout(main_layout)
        
        def load_kline_data(self):
            """加載 K 線數據"""
            if self.load_thread is not None and self.load_thread.isRunning():
                QMessageBox.warning(self, "警告", "數據加載中，請稍候")
                return
            
            symbol = self.symbol_combo.currentText()
            timeframe = self.timeframe_combo.currentText()
            
            # 顯示進度條
            self.progress_bar.setVisible(True)
            self.load_btn.setEnabled(False)
            self.clear_cache_btn.setEnabled(False)
            self.info_text.setText(f"正在加載 {symbol} {timeframe} 數據...\n")
            
            # 創建后台線程
            self.load_thread = DataLoadThread(self.fetcher, symbol, timeframe)
            self.load_thread.data_loaded.connect(self.on_data_loaded)
            self.load_thread.error.connect(self.on_error)
            self.load_thread.finished.connect(self.on_finished)
            self.load_thread.start()
        
        def on_data_loaded(self, df, stats):
            """數據加載完成回調"""
            self.current_data = df
            
            # 更新信息面板
            info_text = f"""幣種: {self.symbol_combo.currentText()}
時間框架: {self.timeframe_combo.currentText()}
總記錄數: {stats['total_records']}
時間範圍: {stats['start_time']} 至 {stats['end_time']}
當前價格: ${stats['current_price']:.2f}
24H 高: ${stats['high_24h']:.2f}
24H 低: ${stats['low_24h']:.2f}"""
            
            self.info_text.setText(info_text)
            
            # 更新表格
            self.update_table(df)
        
        def on_error(self, error_msg):
            """加載出錯回調"""
            self.info_text.setText(f"錯誤: {error_msg}")
            QMessageBox.critical(self, "錯誤", error_msg)
        
        def on_finished(self):
            """加載完成回調"""
            self.progress_bar.setVisible(False)
            self.load_btn.setEnabled(True)
            self.clear_cache_btn.setEnabled(True)
        
        def update_table(self, df):
            """更新表格"""
            self.table.setRowCount(0)
            
            # 只顯示最后 100 條記錄
            df_display = df.tail(100)
            
            for idx, row in df_display.iterrows():
                row_position = self.table.rowCount()
                self.table.insertRow(row_position)
                
                # 時間
                time_str = row['time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in row and pd.notna(row['time']) else "N/A"
                self.table.setItem(row_position, 0, QTableWidgetItem(time_str))
                
                # 開盤
                open_price = f"{row['open']:.2f}" if 'open' in row and pd.notna(row['open']) else "N/A"
                self.table.setItem(row_position, 1, QTableWidgetItem(open_price))
                
                # 最高
                high_price = f"{row['high']:.2f}" if 'high' in row and pd.notna(row['high']) else "N/A"
                self.table.setItem(row_position, 2, QTableWidgetItem(high_price))
                
                # 最低
                low_price = f"{row['low']:.2f}" if 'low' in row and pd.notna(row['low']) else "N/A"
                self.table.setItem(row_position, 3, QTableWidgetItem(low_price))
                
                # 收盤
                close_price = f"{row['close']:.2f}" if 'close' in row and pd.notna(row['close']) else "N/A"
                self.table.setItem(row_position, 4, QTableWidgetItem(close_price))
                
                # 成交量
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

else:
    # ==================== Tkinter 版本 (備選方案) ====================
    import tkinter as tk
    from tkinter import ttk, messagebox
    import threading
    
    class ModelEnsembleGUI(tk.Tk):
        """使用 Tkinter 的 GUI 界面"""
        
        def __init__(self):
            super().__init__()
            self.title("加密貨幣 K 線數據查詢系統")
            self.geometry("1200x800")
            
            self.fetcher = KlineDataFetcher()
            self.current_data = None
            
            self.init_ui()
        
        def init_ui(self):
            """初始化 UI"""
            # 控制面板
            control_frame = ttk.Frame(self, padding="10")
            control_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Label(control_frame, text="選擇幣種:").grid(row=0, column=0, padx=5)
            self.symbol_var = tk.StringVar(value="BTCUSDT")
            symbol_combo = ttk.Combobox(
                control_frame,
                textvariable=self.symbol_var,
                values=SUPPORTED_SYMBOLS,
                width=15,
                state="readonly"
            )
            symbol_combo.grid(row=0, column=1, padx=5)
            
            ttk.Label(control_frame, text="時間框架:").grid(row=0, column=2, padx=5)
            self.timeframe_var = tk.StringVar(value="15m")
            timeframe_combo = ttk.Combobox(
                control_frame,
                textvariable=self.timeframe_var,
                values=TIMEFRAMES,
                width=10,
                state="readonly"
            )
            timeframe_combo.grid(row=0, column=3, padx=5)
            
            ttk.Button(
                control_frame,
                text="加載數據",
                command=self.load_kline_data
            ).grid(row=0, column=4, padx=5)
            
            ttk.Button(
                control_frame,
                text="清空緩存",
                command=self.clear_cache
            ).grid(row=0, column=5, padx=5)
            
            # 信息面板
            info_frame = ttk.LabelFrame(self, text="數據信息", padding="10")
            info_frame.pack(fill=tk.X, padx=10, pady=10)
            
            self.info_text = tk.Text(info_frame, height=6, width=80)
            self.info_text.pack(fill=tk.BOTH, expand=True)
            
            # 數據表格框
            table_frame = ttk.LabelFrame(self, text="K 線數據", padding="10")
            table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            scrollbar = ttk.Scrollbar(table_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.tree = ttk.Treeview(
                table_frame,
                columns=("時間", "開盤", "最高", "最低", "收盤", "成交量"),
                height=15,
                yscrollcommand=scrollbar.set
            )
            scrollbar.config(command=self.tree.yview)
            
            self.tree.column("#0", width=0, stretch=tk.NO)
            self.tree.column("時間", anchor=tk.CENTER, width=180)
            self.tree.column("開盤", anchor=tk.E, width=100)
            self.tree.column("最高", anchor=tk.E, width=100)
            self.tree.column("最低", anchor=tk.E, width=100)
            self.tree.column("收盤", anchor=tk.E, width=100)
            self.tree.column("成交量", anchor=tk.E, width=150)
            
            self.tree.heading("#0", text="")
            self.tree.heading("時間", text="時間")
            self.tree.heading("開盤", text="開盤")
            self.tree.heading("最高", text="最高")
            self.tree.heading("最低", text="最低")
            self.tree.heading("收盤", text="收盤")
            self.tree.heading("成交量", text="成交量")
            
            self.tree.pack(fill=tk.BOTH, expand=True)
        
        def load_kline_data(self):
            """加載 K 線數據"""
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, f"正在加載 {symbol} {timeframe} 數據...\n")
            
            # 在後台線程加載
            thread = threading.Thread(
                target=self._load_data_thread,
                args=(symbol, timeframe)
            )
            thread.daemon = True
            thread.start()
        
        def _load_data_thread(self, symbol, timeframe):
            """后台加載線程"""
            try:
                df = self.fetcher.fetch_kline_data(symbol, timeframe)
                
                if df is None or len(df) == 0:
                    self.info_text.config(state=tk.NORMAL)
                    self.info_text.insert(tk.END, "數據加載失敗！\n")
                    self.info_text.config(state=tk.DISABLED)
                    messagebox.showerror("錯誤", f"無法加載 {symbol} {timeframe} 的數據")
                    return
                
                # 更新信息
                info_text = f"""幣種: {symbol}
時間框架: {timeframe}
總記錄數: {len(df)}"""
                
                if 'time' in df.columns:
                    info_text += f"\n時間範圍: {df['time'].min()} 至 {df['time'].max()}"
                
                if 'close' in df.columns:
                    info_text += f"\n當前價格: ${df['close'].iloc[-1]:.2f}"
                
                self.info_text.config(state=tk.NORMAL)
                self.info_text.delete("1.0", tk.END)
                self.info_text.insert(tk.END, info_text)
                self.info_text.config(state=tk.DISABLED)
                
                # 更新表格
                self._update_table(df)
                
            except Exception as e:
                logger.error(f"加載數據出錯: {str(e)}")
                self.info_text.config(state=tk.NORMAL)
                self.info_text.insert(tk.END, f"出錯: {str(e)}\n")
                self.info_text.config(state=tk.DISABLED)
                messagebox.showerror("錯誤", f"加載數據出錯: {str(e)}")
        
        def _update_table(self, df):
            """更新表格"""
            # 清空表格
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 只顯示最后 100 條記錄
            df_display = df.tail(100)
            
            for idx, row in df_display.iterrows():
                time_str = row['time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in row and pd.notna(row['time']) else "N/A"
                open_price = f"{row['open']:.2f}" if 'open' in row and pd.notna(row['open']) else "N/A"
                high_price = f"{row['high']:.2f}" if 'high' in row and pd.notna(row['high']) else "N/A"
                low_price = f"{row['low']:.2f}" if 'low' in row and pd.notna(row['low']) else "N/A"
                close_price = f"{row['close']:.2f}" if 'close' in row and pd.notna(row['close']) else "N/A"
                volume = f"{row['volume']:.0f}" if 'volume' in row and pd.notna(row['volume']) else "N/A"
                
                self.tree.insert(
                    "",
                    tk.END,
                    values=(time_str, open_price, high_price, low_price, close_price, volume)
                )
        
        def clear_cache(self):
            """清空緩存"""
            self.fetcher.clear_cache()
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, "已清空所有緩存")
            self.info_text.config(state=tk.DISABLED)
            messagebox.showinfo("成功", "已清空所有緩存")
    
    
    def main():
        app = ModelEnsembleGUI()
        app.mainloop()


if __name__ == "__main__":
    main()
