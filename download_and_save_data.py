#!/usr/bin/env python3
"""
明確下載並保存数據到本地

步驟:
1. 從 Hugging Face 下載 BTC 15m 數據
2. 保存到 ./data/ 資料夾
3. 驗證數據完整性
4. 列出本地文件
"""

import os
import sys
from datetime import datetime

print("\n" + "="*70)
print("明確下載並保存数據")
print("="*70)

# 檢查當前工作路徑
print(f"\n當前工作路徑: {os.getcwd()}")

# Step 1: 檢查依賴
print("\n[Step 1] 檢查依賴...")
try:
    import pandas as pd
    import numpy as np
    from huggingface_hub import hf_hub_download
    print("  依賴正常")
except ImportError as e:
    print(f"  依賴缺失: {e}")
    sys.exit(1)

# Step 2: 樣本外 Token
print("\n[Step 2] 樣本外 HF_TOKEN...")
try:
    from config import HF_TOKEN
    if not HF_TOKEN or HF_TOKEN == "your_token_here":
        print("  HF_TOKEN 未設定")
        sys.exit(1)
    print(f"  Token 已設定: {HF_TOKEN[:10]}...{HF_TOKEN[-5:]}")
except Exception as e:
    print(f"  錯誤: {e}")
    sys.exit(1)

# Step 3: 統一本地数據路徑
print("\n[Step 3] 準備本地數據路徑...")
LOCAL_DATA_DIR = "./data"
LOCAL_FILE_PATH = os.path.join(LOCAL_DATA_DIR, "btc_15m.parquet")

if not os.path.exists(LOCAL_DATA_DIR):
    os.makedirs(LOCAL_DATA_DIR)
    print(f"  已建立料夾夾: {LOCAL_DATA_DIR}")
else:
    print(f"  料夾夾已存在: {LOCAL_DATA_DIR}")

print(f"  文件保存路徑: {os.path.abspath(LOCAL_FILE_PATH)}")

# Step 4: 檢查是否已有本地文件
print("\n[Step 4] 檢查是否已有本地文件...")
if os.path.exists(LOCAL_FILE_PATH):
    file_size = os.path.getsize(LOCAL_FILE_PATH) / (1024*1024)
    print(f"  本地文件已存在: {LOCAL_FILE_PATH}")
    print(f"  文件大小: {file_size:.2f} MB")
    df = pd.read_parquet(LOCAL_FILE_PATH)
    print(f"  數據形狀: {df.shape}")
else:
    print(f"  本地文件不存在，正從遠端下載...")
    
    # Step 5: 下載數據
    print("\n[Step 5] 從 Hugging Face 下載...")
    try:
        file_path = hf_hub_download(
            repo_id="zongowo111/v2-crypto-ohlcv-data",
            filename="klines/BTCUSDT/BTC_15m.parquet",
            repo_type="dataset",
            token=HF_TOKEN,
            cache_dir="./cache"
        )
        print(f"  下載完成: {file_path}")
        print(f"  下載文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        # Step 6: 載入數據
        print("\n[Step 6] 載入數據...")
        df = pd.read_parquet(file_path)
        print(f"  數據形狀: {df.shape}")
        print(f"  敘章: {df.columns.tolist()}")
        
        # Step 7: 清理数據
        print("\n[Step 7] 清理数據...")
        
        # 处理 timestamp 列
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            print("  已設置 timestamp 為索引")
        
        # 刪除 symbol 列
        if 'symbol' in df.columns:
            df = df.drop('symbol', axis=1)
            print("  已刪除 symbol 列")
        
        df.index.name = 'timestamp'
        df = df.sort_index()
        print(f"  清理後數據形狀: {df.shape}")
        
        # Step 8: 保存到本地
        print("\n[Step 8] 保存到本地...")
        print(f"  正在保存到: {LOCAL_FILE_PATH}")
        
        df.to_parquet(LOCAL_FILE_PATH)
        
        # 驗證保存
        if os.path.exists(LOCAL_FILE_PATH):
            file_size = os.path.getsize(LOCAL_FILE_PATH) / (1024*1024)
            print(f"  保存成功")
            print(f"  文件大小: {file_size:.2f} MB")
            print(f"  文件位置: {os.path.abspath(LOCAL_FILE_PATH)}")
        else:
            print(f"  錯誤: 保存失敗")
            sys.exit(1)
        
    except Exception as e:
        print(f"  下載失敗: {e}")
        sys.exit(1)

# Step 9: 最終驗證
print("\n[Step 9] 最終驗證...")
if os.path.exists(LOCAL_FILE_PATH):
    df = pd.read_parquet(LOCAL_FILE_PATH)
    print(f"  文件存在: 是")
    print(f"  數據形狀: {df.shape}")
    print(f"  時間範圍: {df.index[0]} ~ {df.index[-1]}")
    print(f"\n  第一行:")
    print(df.head(1))
else:
    print(f"  錯誤: 文件不存在")
    sys.exit(1)

print("\n" + "="*70)
print("數據下載並保存成功")
print("="*70)
print(f"\n下一步: python run_optimization.py\n")
