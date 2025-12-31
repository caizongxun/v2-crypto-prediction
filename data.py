"""
數據加載模塊

支持從 Hugging Face 數據集加載 BTC OHLCV 數據
數據結構：
  klines/BTCUSDT/BTC_15m.parquet
  klines/BTCUSDT/BTC_1h.parquet
  klines/{SYMBOL}/{SYMBOL}_15m.parquet
"""

import pandas as pd
import numpy as np
from typing import Optional
import os
from datetime import datetime


def load_btc_data(
    hf_token: str,
    timeframe: str = "15m",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    repo_id: str = "zongowo111/v2-crypto-ohlcv-data"
) -> Optional[pd.DataFrame]:
    """
    從 Hugging Face 數據集加載 BTC OHLCV 數據
    
    數據結構：
      huggingface.co/datasets/{repo_id}/blob/main/klines/BTCUSDT/BTC_{timeframe}.parquet
    
    Args:
        hf_token: Hugging Face API token
        timeframe: 時間框 ("15m" 或 "1h")
        start_date: 開始日期 (YYYY-MM-DD)
        end_date: 結束日期 (YYYY-MM-DD)
        repo_id: Hugging Face 數據集 ID
    
    Returns:
        pd.DataFrame: OHLCV 數據 (帶時間索引)
        None: 如果加載失敗
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # 文件路徑：klines/BTCUSDT/BTC_15m.parquet
        file_path = f"klines/BTCUSDT/BTC_{timeframe}.parquet"
        
        print(f"  正在從 Hugging Face 下載: {file_path}")
        print(f"  Repo: {repo_id}")
        
        # 下載文件
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            token=hf_token,
            cache_dir="./cache"
        )
        
        print(f"  下載完成: {local_path}")
        
        # 加載 Parquet 文件
        df = pd.read_parquet(local_path)
        
        print(f"  原始數據形狀: {df.shape}")
        print(f"  原始列名: {df.columns.tolist()}")
        
        # 驗證必需的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print(f"  警告：缺少列: {[c for c in required_columns if c not in df.columns]}")
            print(f"  可用的列: {df.columns.tolist()}")
            return None
        
        # 處理時間列
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif df.index.name == 'timestamp':
            df.index = pd.to_datetime(df.index)
        else:
            # 嘗試找到時間列
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                df.set_index(time_cols[0], inplace=True)
                df.index = pd.to_datetime(df.index)
            else:
                print("  警告：找不到時間列，使用數據的索引")
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
        
        # 確保索引名稱
        df.index.name = 'timestamp'
        
        # 排序
        df = df.sort_index()
        
        print(f"  處理後數據形狀: {df.shape}")
        print(f"  時間範圍: {df.index[0]} ~ {df.index[-1]}")
        
        # 應用日期過濾
        if start_date:
            start_dt = pd.to_datetime(start_date)
            print(f"  過濾開始日期: {start_date}")
            df = df[df.index >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            # 包含整個結束日期
            end_dt = pd.to_datetime(end_date + ' 23:59:59')
            print(f"  過濾結束日期: {end_date}")
            df = df[df.index <= end_dt]
        
        print(f"  最終數據形狀: {df.shape}")
        print(f"  最終時間範圍: {df.index[0]} ~ {df.index[-1]}")
        
        if len(df) == 0:
            print("  警告：過濾後沒有數據")
            return None
        
        return df
        
    except ImportError:
        print("  錯誤: huggingface_hub 未安裝")
        print("     解決方案: pip install huggingface_hub")
        return None
    
    except Exception as e:
        print(f"  錯誤: {str(e)}")
        print(f"\n  排查步驟:")
        print(f"  1. 檢查 HF_TOKEN 是否有效")
        print(f"     Token: {hf_token[:10]}...{hf_token[-5:] if len(hf_token) > 15 else ''}")
        print(f"  2. 驗證是否可以訪問數據集:")
        print(f"     https://huggingface.co/datasets/{repo_id}")
        print(f"  3. 檢查文件是否存在:")
        print(f"     {repo_id}/klines/BTCUSDT/BTC_{timeframe}.parquet")
        return None


def load_crypto_data(
    symbol: str,
    hf_token: str,
    timeframe: str = "15m",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    repo_id: str = "zongowo111/v2-crypto-ohlcv-data"
) -> Optional[pd.DataFrame]:
    """
    從 Hugging Face 數據集加載任何加密貨幣 OHLCV 數據
    
    數據結構：
      huggingface.co/datasets/{repo_id}/blob/main/klines/{SYMBOL}/{SYMBOL}_{timeframe}.parquet
    
    Args:
        symbol: 交易對 (e.g., "BTCUSDT", "ETHUSDT")
        hf_token: Hugging Face API token
        timeframe: 時間框 ("15m" 或 "1h")
        start_date: 開始日期 (YYYY-MM-DD)
        end_date: 結束日期 (YYYY-MM-DD)
        repo_id: Hugging Face 數據集 ID
    
    Returns:
        pd.DataFrame: OHLCV 數據
        None: 如果加載失敗
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # 輸取最後對的前三字母作為文件名
        # e.g., BTCUSDT -> BTC_15m.parquet
        coin = symbol[:3] if len(symbol) >= 3 else symbol
        file_path = f"klines/{symbol}/{coin}_{timeframe}.parquet"
        
        print(f"  正在從 Hugging Face 下載: {file_path}")
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            token=hf_token,
            cache_dir="./cache"
        )
        
        print(f"  下載完成")
        
        # 加載數據
        df = pd.read_parquet(local_path)
        
        # 驗證必需的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return None
        
        # 處理時間索引
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                df.set_index(time_cols[0], inplace=True)
            df.index = pd.to_datetime(df.index)
        
        df.index.name = 'timestamp'
        
        # 排序
        df = df.sort_index()
        
        # 日期過濾
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date + ' 23:59:59']
        
        print(f"  數據條數: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"  錯誤: {str(e)}")
        return None


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    驗證 OHLCV 數據的有效性
    
    Args:
        df: OHLCV 數據框
    
    Returns:
        bool: 是否有效
    """
    # 驗證需要的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"警告: 缺少列: {[c for c in required_cols if c not in df.columns]}")
        return False
    
    # 驗證數據類型
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"警告: {col} 不是數值類型")
            return False
    
    # 驗證完整性
    if df.isnull().any().any():
        print(f"警告: 存在 NULL 值")
        return False
    
    # OHLC 邏輯（H >= L, H >= O, H >= C, L <= O, L <= C)
    if not (df['high'] >= df['low']).all():
        print(f"警告: High < Low 的 K 線")
        return False
    
    return True


if __name__ == "__main__":
    # 測試腳本
    from config import HF_TOKEN
    
    print("\n" + "="*70)
    print("數據加載測試")
    print("="*70)
    
    print("\n[測試 1] 加載 BTC 15m 數據")
    df = load_btc_data(hf_token=HF_TOKEN, start_date='2024-01-01', end_date='2024-12-31')
    if df is not None:
        print(f"\n  數據形狀: {df.shape}")
        print(f"  列名: {df.columns.tolist()}")
        print(f"\n  截斷：")
        print(df.head())
    
    print("\n" + "="*70)
