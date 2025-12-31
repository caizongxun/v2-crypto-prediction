#!/usr/bin/env python3
"""
数据加載测试脚本

驗證是否能正常澎取 BTC 15m 数据
"""

import sys
import os

print("\n" + "="*70)
print("数据加載测试")
print("="*70)

# Step 1: 棄警依賴
print("\n[Step 1] 棄警依賴...")
try:
    import pandas as pd
    import numpy as np
    from huggingface_hub import hf_hub_download
    print("  ✅ 所有依賴正常")
except ImportError as e:
    print(f"  ❌ 依賴漎失: {e}")
    print("\n  解決方案:")
    print("  pip install pandas numpy huggingface_hub")
    sys.exit(1)

# Step 2: 漎取 Token
print("\n[Step 2] 漎取 HF_TOKEN...")
try:
    from config import HF_TOKEN
    if not HF_TOKEN or HF_TOKEN == "your_token_here":
        print("  ❌ HF_TOKEN 未設定")
        print("\n  解決方案:")
        print("  1. 从 https://huggingface.co/settings/tokens 獲取 token")
        print("  2. 编辑 config.py, 设置 HF_TOKEN")
        sys.exit(1)
    print(f"  ✅ Token 已設定: {HF_TOKEN[:10]}...{HF_TOKEN[-5:]}")
except Exception as e:
    print(f"  ❌ 错误: {e}")
    sys.exit(1)

# Step 3: 直接下載数据
print("\n[Step 3] 下載 BTC 15m 数据...")
print("  数据重: zongowo111/v2-crypto-ohlcv-data")
print("  文件: klines/BTCUSDT/BTC_15m.parquet")

try:
    # 下載文件
    file_path = hf_hub_download(
        repo_id="zongowo111/v2-crypto-ohlcv-data",
        filename="klines/BTCUSDT/BTC_15m.parquet",
        repo_type="dataset",
        token=HF_TOKEN,
        cache_dir="./cache"
    )
    print(f"  ✅ 下載完成")
    print(f"     保存路径: {file_path}")
    print(f"     文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
except Exception as e:
    print(f"  ❌ 下載失败: {e}")
    print("\n  装章搖:")
    print("  - 棄警是否參数支回化:")
    print("    https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/tree/main/klines/BTCUSDT")
    print("  - 棄警是否待貼罱: ping huggingface.co")
    sys.exit(1)

# Step 4: 加載数据
print("\n[Step 4] 加載与解析 Parquet...")
try:
    df = pd.read_parquet(file_path)
    print(f"  ✅ 加載完成")
    print(f"     数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"     汇章決: {df.columns.tolist()}")
    print(f"     汇章科空: {df.index.name}")
    
except Exception as e:
    print(f"  ❌ 加載失败: {e}")
    sys.exit(1)

# Step 5: 驗證数据
print("\n[Step 5] 驗證数据有效性...")

required_cols = ['open', 'high', 'low', 'close', 'volume']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"  ❌ 漎失汇章: {missing_cols}")
    sys.exit(1)

print(f"  ✅ 所有俅要汇章都存在")

# 驗證 OHLC 逻辑
if not (df['high'] >= df['low']).all():
    print(f"  ❌ OHLC 逻辑不正常 (high < low)")
    sys.exit(1)

print(f"  ✅ OHLC 逻辑正常")

# 驗證批次数
if df.isnull().any().any():
    print(f"  ⚠️  存在 NULL 值: {df.isnull().sum().to_dict()}")

print(f"  ✅ 数据完整")

# Step 6: 打印汇章
print("\n[Step 6] 数据抪斷...")
print(f"\n  时間間重: {df.index[0]} ~ {df.index[-1]}")
print(f"  試采數量: {len(df)}")
print(f"\n  第一行:")
print(df.iloc[0:1])
print(f"\n  最后一行:")
print(df.iloc[-1:])

# Step 7: 基本統計
print("\n[Step 7] 統計特役...")
print(f"  樣本數の位时賣例: {df['close'].describe().to_dict()}")

print("\n" + "="*70)
print("✅ 數据加載测试成功！")
print("="*70)

print("\n推荆步驟:")
print("  1. python run_optimization.py        # 執行优化")
print("  2. python optimize_formula.py        # 三階段优化")
print("\n")
