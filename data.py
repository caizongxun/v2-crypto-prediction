"""
数据加載模块

支持从 Hugging Face 数据集加載 BTC OHLCV 数据
数据结构：
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
    从 Hugging Face 数据集加載 BTC OHLCV 数据
    
    数据结构：
      huggingface.co/datasets/{repo_id}/blob/main/klines/BTCUSDT/BTC_{timeframe}.parquet
    
    Args:
        hf_token: Hugging Face API token
        timeframe: 時間框 ("15m" 或 "1h")
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        repo_id: Hugging Face 数据集 ID
    
    Returns:
        pd.DataFrame: OHLCV 数据 (带时间索引)
        None: 如果加載失败
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # 文件路径： klines/BTCUSDT/BTC_15m.parquet
        file_path = f"klines/BTCUSDT/BTC_{timeframe}.parquet"
        
        print(f"  正在从 Hugging Face 下載: {file_path}")
        print(f"  Repo: {repo_id}")
        print(f"  Timeframe: {timeframe}")
        
        # 下載文件
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            token=hf_token,
            cache_dir="./cache"
        )
        
        print(f"  ✅ 下載完成: {local_path}")
        
        # 加載 Parquet 文件
        df = pd.read_parquet(local_path)
        
        # 驗證必需的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print(f"  ⚠⚠ 警告：缺少列: {[c for c in required_columns if c not in df.columns]}")
            print(f"  宜有的列: {df.columns.tolist()}")
            return None
        
        # 确保星期数时标
        if df.index.name != 'timestamp':
            # 尝试找时间列
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                df.set_index(time_cols[0], inplace=True)
                df.index.name = 'timestamp'
            else:
                print("  ⚠⚠ 警告：找不到时间列，会使用第一列作为索引")
        
        # 确保索引是时间类类
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 对排序
        df = df.sort_index()
        
        # 应用日期过滤
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        print(f"  数据条数: {len(df)}")
        print(f"  时间范围: {df.index[0]} ~ {df.index[-1]}")
        
        return df
        
    except ImportError:
        print("  ❌ 错误: huggingface_hub 未安装")
        print("     設置方案: pip install huggingface_hub")
        return None
    
    except Exception as e:
        print(f"  ❌ 错误: {str(e)}")
        print(f"\n  排鮳步驟:")
        print(f"  1. 検查 HF_TOKEN 是否有效")
        print(f"     Token: {hf_token[:10]}...{hf_token[-5:]}")
        print(f"  2. 碩棄是否可以訪啊數據集:")
        print(f"     https://huggingface.co/datasets/{repo_id}")
        print(f"  3. 検查文件是否存在:")
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
    从 Hugging Face 数据集加載任何加密货币 OHLCV 数据
    
    数据结构：
      huggingface.co/datasets/{repo_id}/blob/main/klines/{SYMBOL}/{SYMBOL}_{timeframe}.parquet
    
    Args:
        symbol: 交易对 (e.g., "BTCUSDT", "ETHUSDT")
        hf_token: Hugging Face API token
        timeframe: 時間框 ("15m" 或 "1h")
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        repo_id: Hugging Face 数据集 ID
    
    Returns:
        pd.DataFrame: OHLCV 数据
        None: 如果加載失败
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # 输取最后对的前三字母作为文件名
        # e.g., BTCUSDT -> BTC_15m.parquet
        coin = symbol[:3] if len(symbol) >= 3 else symbol
        file_path = f"klines/{symbol}/{coin}_{timeframe}.parquet"
        
        print(f"  正在从 Hugging Face 下載: {file_path}")
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            token=hf_token,
            cache_dir="./cache"
        )
        
        print(f"  ✅ 下載完成")
        
        # 加載数据
        df = pd.read_parquet(local_path)
        
        # 驗證必需的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return None
        
        # 驗證索引
        if not isinstance(df.index, pd.DatetimeIndex):
            if df.index.name == 'timestamp':
                df.index = pd.to_datetime(df.index)
            else:
                time_cols = [col for col in df.columns if 'time' in col.lower()]
                if time_cols:
                    df.set_index(time_cols[0], inplace=True)
                    df.index = pd.to_datetime(df.index)
        
        # 排序
        df = df.sort_index()
        
        # 日期过滤
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        print(f"  数据条数: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"  ❌ 错误: {str(e)}")
        return None


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    驗證 OHLCV 数据的有效性
    
    Args:
        df: OHLCV 数据框
    
    Returns:
        bool: 是否有效
    """
    # 棄警需要的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"⚠⚠ 缺少列: {[c for c in required_cols if c not in df.columns]}")
        return False
    
    # 棄警数据类型
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"⚠⚠ {col} 不是数值类型")
            return False
    
    # 棄警整整性
    if df.isnull().any().any():
        print(f"⚠⚠ 存在 NULL 值")
        return False
    
    # OHLC 逻辑（H >= L, H >= O, H >= C, L <= O, L <= C)
    if not (df['high'] >= df['low']).all():
        print(f"⚠⚠ High < Low 的 K 線")
        return False
    
    return True


if __name__ == "__main__":
    # 测试脚本
    from config import HF_TOKEN
    
    print("\n" + "="*70)
    print("数据加載测试")
    print("="*70)
    
    print("\n[测试 1] 加載 BTC 15m 数据")
    df = load_btc_data(hf_token=HF_TOKEN, start_date='2024-01-01', end_date='2024-12-31')
    if df is not None:
        print(f"\n  数据形状: {df.shape}")
        print(f"  列名: {df.columns.tolist()}")
        print(f"\n  抪断：")
        print(df.head())
    
    print("\n" + "="*70)
