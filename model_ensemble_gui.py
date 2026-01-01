import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
import logging
import threading

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HuggingFace 數據集配置
HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
HF_REPO_TYPE = "dataset"

# 支持的幣種列表
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
            symbol_prefix = symbol.replace("USDT", "")
            filename = f"klines/{symbol}/{symbol_prefix}_{timeframe}.parquet"
            
            logger.info(f"正在從 HuggingFace 下載: {filename}")
            
            # 從 HuggingFace 下載文件
            file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type=HF_REPO_TYPE,
                force_download=False
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


class KlineGUI:
    """使用 tkinter 的 GUI 界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("加密貨幣 K 線數據查詢系統")
        self.root.geometry("1200x800")
        
        self.fetcher = KlineDataFetcher()
        self.current_data = None
        self.loading = False
        
        self.init_ui()
    
    def init_ui(self):
        """初始化 UI"""
        # 頂部控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # 幣種選擇
        ttk.Label(control_frame, text="選擇幣種:").pack(side=tk.LEFT, padx=5)
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        symbol_combo = ttk.Combobox(control_frame, textvariable=self.symbol_var, 
                                     values=SUPPORTED_SYMBOLS, state="readonly", width=15)
        symbol_combo.pack(side=tk.LEFT, padx=5)
        
        # 時間框架選擇
        ttk.Label(control_frame, text="時間框架:").pack(side=tk.LEFT, padx=5)
        self.timeframe_var = tk.StringVar(value="15m")
        timeframe_combo = ttk.Combobox(control_frame, textvariable=self.timeframe_var,
                                        values=TIMEFRAMES, state="readonly", width=10)
        timeframe_combo.pack(side=tk.LEFT, padx=5)
        
        # 加載按鈕
        self.load_btn = ttk.Button(control_frame, text="加載數據", command=self.load_kline_data)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # 清空緩存按鈕
        self.clear_cache_btn = ttk.Button(control_frame, text="清空緩存", command=self.clear_cache)
        self.clear_cache_btn.pack(side=tk.LEFT, padx=5)
        
        # 進度條
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, 
                                             maximum=100, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # 信息面板
        info_frame = ttk.LabelFrame(self.root, text="數據信息", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=5, width=100)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.config(state=tk.DISABLED)
        
        # 表格框架
        table_frame = ttk.LabelFrame(self.root, text="K 線數據", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 創建 Treeview 表格
        columns = ("時間", "開盤", "最高", "最低", "收盤", "成交量")
        self.tree = ttk.Treeview(table_frame, columns=columns, height=25)
        self.tree.column("#0", width=0, stretch=tk.NO)
        
        for col in columns:
            self.tree.column(col, anchor=tk.CENTER, width=180)
            self.tree.heading(col, text=col)
        
        # 滾動條
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_kline_data(self):
        """加載 K 線數據（後台線程）"""
        if self.loading:
            messagebox.showwarning("警告", "數據加載中，請稍候")
            return
        
        symbol = self.symbol_var.get()
        timeframe = self.timeframe_var.get()
        
        # 禁用按鈕
        self.load_btn.config(state=tk.DISABLED)
        self.clear_cache_btn.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.loading = True
        
        # 更新信息
        self.update_info_text(f"正在加載 {symbol} {timeframe} 數據...\n")
        
        # 在後台線程中執行
        thread = threading.Thread(target=self._load_data_thread, args=(symbol, timeframe))
        thread.daemon = True
        thread.start()
    
    def _load_data_thread(self, symbol, timeframe):
        """後台數據加載線程"""
        try:
            df = self.fetcher.fetch_kline_data(symbol, timeframe)
            
            if df is None or len(df) == 0:
                self.root.after(0, lambda: messagebox.showerror("錯誤", "無法加載數據或數據為空"))
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
            
            # 在主線程中更新 UI
            self.root.after(0, self._update_ui, df, stats)
            
        except Exception as e:
            logger.error(f"加載數據出錯: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("錯誤", f"加載數據出錯: {str(e)}"))
        
        finally:
            self.loading = False
            self.root.after(0, self._on_load_finished)
    
    def _update_ui(self, df, stats):
        """更新 UI"""
        self.current_data = df
        
        # 更新信息面板
        info_text = f"""幣種: {self.symbol_var.get()}
時間框架: {self.timeframe_var.get()}
總記錄數: {stats['total_records']}
時間範圍: {stats['start_time']} 至 {stats['end_time']}
當前價格: ${stats['current_price']:.2f}
24H 高: ${stats['high_24h']:.2f}
24H 低: ${stats['low_24h']:.2f}"""
        
        self.update_info_text(info_text)
        
        # 更新表格
        self.update_table(df)
    
    def _on_load_finished(self):
        """加載完成回調"""
        self.progress_bar.stop()
        self.load_btn.config(state=tk.NORMAL)
        self.clear_cache_btn.config(state=tk.NORMAL)
    
    def update_info_text(self, text):
        """更新信息文本"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        self.info_text.config(state=tk.DISABLED)
    
    def update_table(self, df):
        """更新表格"""
        # 清空表格
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 只顯示最後 100 條記錄
        df_display = df.tail(100)
        
        for idx, row in df_display.iterrows():
            time_str = row['time'].strftime('%Y-%m-%d %H:%M:%S') if 'time' in row and pd.notna(row['time']) else "N/A"
            open_price = f"{row['open']:.2f}" if 'open' in row and pd.notna(row['open']) else "N/A"
            high_price = f"{row['high']:.2f}" if 'high' in row and pd.notna(row['high']) else "N/A"
            low_price = f"{row['low']:.2f}" if 'low' in row and pd.notna(row['low']) else "N/A"
            close_price = f"{row['close']:.2f}" if 'close' in row and pd.notna(row['close']) else "N/A"
            volume = f"{row['volume']:.0f}" if 'volume' in row and pd.notna(row['volume']) else "N/A"
            
            self.tree.insert("", tk.END, values=(time_str, open_price, high_price, low_price, close_price, volume))
    
    def clear_cache(self):
        """清空緩存"""
        self.fetcher.clear_cache()
        self.update_info_text("已清空所有緩存")
        messagebox.showinfo("成功", "已清空所有緩存")


def main():
    root = tk.Tk()
    gui = KlineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
