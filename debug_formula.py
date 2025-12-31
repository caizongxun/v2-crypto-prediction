#!/usr/bin/env python3
"""
公式詳標除錯腳本

詳謁每一步的信號生成機制
"""

import os
import sys
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("公式詳標除錯")
print("="*70)

# Step 1: 加載數據
print("\n[Step 1] 加載數據...")
try:
    df = pd.read_parquet("./data/btc_15m.parquet")
    
    # 過濾 2024 數據
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-12-31 23:59:59')
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    print(f"  数据形狀: {df.shape}")
except Exception as e:
    print(f"  錯誤: {e}")
    sys.exit(1)

# Step 2: 加載技術指標
print("\n[Step 2] 計算技術指標...")
try:
    from formulas.indicators import IndicatorCalculator
    
    calc = IndicatorCalculator(df)
    df = calc.calculate_all_indicators()
    print(f"  技術指標計算完成")
    print(f"  托欄: {df.columns.tolist()}")
except Exception as e:
    print(f"  錯誤: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: 棄警技術指標
print("\n[Step 3] 棄警技術指標...")
try:
    # 檢查是否有 NaN 值
    nan_counts = df.isnull().sum()
    print(f"  NaN 計整:")
    for col in ['EMA_15', 'EMA_60', 'RSI', 'ATR', 'SuperTrend', 'ADX']:
        if col in df.columns:
            count = nan_counts[col]
            pct = count / len(df) * 100
            print(f"    {col}: {count} ({pct:.1f}%)")
    
    # 検查值範圍
    print(f"\n  值範圍:")
    print(f"    ATR: min={df['ATR'].min():.2f}, max={df['ATR'].max():.2f}, mean={df['ATR'].mean():.2f}")
    print(f"    RSI: min={df['RSI'].min():.2f}, max={df['RSI'].max():.2f}, mean={df['RSI'].mean():.2f}")
    print(f"    ADX: min={df['ADX'].min():.2f}, max={df['ADX'].max():.2f}, mean={df['ADX'].mean():.2f}")
except Exception as e:
    print(f"  錯誤: {e}")
    import traceback
    traceback.print_exc()

# Step 4: 樣本數据検查
print("\n[Step 4] 樣本數据検查...")
print(f"\n  前 5 根 K 線的技術指標:")
sample_cols = ['close', 'ATR', 'RSI', 'ADX', 'EMA_15', 'EMA_60', 'SuperTrend']
for idx in range(60, 65):
    row = df.iloc[idx]
    print(f"\n  K 線 {idx} ({row.name}):")
    print(f"    close={row['close']:.2f}")
    if 'ATR' in df.columns:
        print(f"    ATR={row['ATR']:.4f}")
    if 'RSI' in df.columns:
        print(f"    RSI={row['RSI']:.2f}")
    if 'ADX' in df.columns:
        print(f"    ADX={row['ADX']:.2f}")
    if 'EMA_15' in df.columns:
        print(f"    EMA_15={row['EMA_15']:.2f}")
    if 'EMA_60' in df.columns:
        print(f"    EMA_60={row['EMA_60']:.2f}")
    if 'SuperTrend' in df.columns:
        print(f"    SuperTrend={row['SuperTrend']:.2f}")

# Step 5: 檢查波幅率遮芽
print("\n[Step 5] 検查 ATR 郆値釯...")
try:
    from formulas.golden_formula_v2_config import GoldenFormulaV2Config
    
    config = GoldenFormulaV2Config()
    min_atr_pct = config.volatility_config.min_atr_percent
    
    print(f"  最低 ATR 百分比: {min_atr_pct}%")
    
    # 計算平均 ATR 百分比
    atr_pct = (df['ATR'] / df['close']) * 100
    print(f"  平均 ATR 百分比: {atr_pct.mean():.4f}%")
    print(f"  最小: {atr_pct.min():.4f}%, 最大: {atr_pct.max():.4f}%")
    
    # 算數多少根過渙値
    passed = (atr_pct >= min_atr_pct).sum()
    failed = (atr_pct < min_atr_pct).sum()
    print(f"  通過遮芽: {passed} 根 ({passed/len(df)*100:.1f}%)")
    print(f"  未通過: {failed} 根 ({failed/len(df)*100:.1f}%)")
    
except Exception as e:
    print(f"  錯誤: {e}")
    import traceback
    traceback.print_exc()

# Step 6: 該次使用更實韓的參數
print("\n[Step 6] 嘗試使用不同的參數...")
try:
    from formulas.golden_formula_v2 import GoldenFormulaV2
    from formulas.golden_formula_v2_config import (
        GoldenFormulaV2Config, TrendConfig, MomentumConfig, 
        VolumeConfig, VolatilityConfig
    )
    
    # 方案 1: 降低 ATR 易儭值
    print("\n  䬝案 1: 降低 ATR 遮芽值 (0.5% → 0.05%)...")
    config = GoldenFormulaV2Config()
    config.volatility_config.min_atr_percent = 0.05
    
    formula = GoldenFormulaV2(config)
    patterns, df_analysis = formula.analyze(df)
    
    print(f"    找到 {len(patterns)} 個信號")
    
    if len(patterns) > 0:
        from formulas.golden_formula_v2 import Signal
        buy_count = sum(1 for p in patterns if p.signal == Signal.BUY)
        sell_count = sum(1 for p in patterns if p.signal == Signal.SELL)
        print(f"    買入: {buy_count}, 賣出: {sell_count}")
        if len(patterns) > 0:
            print(f"    平均信心度: {np.mean([p.confidence for p in patterns]):.4f}")
            print(f"\n    前 5 個信號:")
            for i, p in enumerate(patterns[:5]):
                signal_type = "BUY" if p.signal == Signal.BUY else "SELL"
                print(f"      {i+1}. {p.timestamp} - {signal_type} (confidence: {p.confidence:.4f})")
    else:
        print("    仍然沒有信號…")
        
except Exception as e:
    print(f"  錯誤: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70 + "\n")
