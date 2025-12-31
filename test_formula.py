#!/usr/bin/env python3
"""
公式診斷脚本

検查是否能成功產生买賣信號
"""

import os
import sys
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("公式診斷")
print("="*70)

# Step 1: 加載數據
print("\n[Step 1] 加載數據...")
try:
    df = pd.read_parquet("./data/btc_15m.parquet")
    print(f"  數據形狀: {df.shape}")
    print(f"  時間範圍: {df.index[0]} ~ {df.index[-1]}")
    
    # 過濾 2024 數據
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-12-31 23:59:59')
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    print(f"  過濾後: {df.shape}")
except Exception as e:
    print(f"  錯誤: {e}")
    print("  請先運行: python download_and_save_data.py")
    sys.exit(1)

# Step 2: 診斷公式
print("\n[Step 2] 檢查公式模組...")
try:
    from formulas.golden_formula_v2 import GoldenFormulaV2, Signal
    from formulas.golden_formula_v2_config import GoldenFormulaV2Config, TrendConfig, MomentumConfig, VolumeConfig
    print("  公式模組已加載")
except Exception as e:
    print(f"  錯誤: {e}")
    sys.exit(1)

# Step 3: 使用預設配置
print("\n[Step 3] 建立預設配置...")
try:
    trend_config = TrendConfig()
    momentum_config = MomentumConfig()
    volume_config = VolumeConfig()
    
    config = GoldenFormulaV2Config(
        trend_config=trend_config,
        momentum_config=momentum_config,
        volume_config=volume_config
    )
    
    print(f"  配置已建立")
    print(f"  趨勢 EMA: fast={trend_config.fast_ema_period}, slow={trend_config.slow_ema_period}")
    print(f"  RSI: period={momentum_config.rsi_period}, oversold={momentum_config.rsi_oversold}, overbought={momentum_config.rsi_overbought}")
except Exception as e:
    print(f"  錯誤: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: 診斷公式
print("\n[Step 4] 運行公式分析...")
try:
    formula = GoldenFormulaV2(config)
    print(f"  公式已初始化")
    
    # 執行分析
    print(f"  正在分析 {len(df)} 根 K 線...")
    patterns, df_analysis = formula.analyze(df)
    
    print(f"  分析完成")
    print(f"  找到的买賣信號: {len(patterns)} 區")
    
    if len(patterns) == 0:
        print("  警告: 沒有找到任何买賣信號")
        
        # 診斷為什麼
        print("\n  診斷信息:")
        if 'ema_fast' in df_analysis.columns:
            print(f"    - fast EMA 已計算: {df_analysis['ema_fast'].notna().sum()} 根")
        if 'ema_slow' in df_analysis.columns:
            print(f"    - slow EMA 已計算: {df_analysis['ema_slow'].notna().sum()} 根")
        if 'rsi' in df_analysis.columns:
            print(f"    - RSI 已計算: {df_analysis['rsi'].notna().sum()} 根")
        if 'supertrend' in df_analysis.columns:
            print(f"    - SuperTrend 已計算: {df_analysis['supertrend'].notna().sum()} 根")
        
        # 棄警是否有計算錯誤
        if 'error' in df_analysis.columns:
            errors = df_analysis[df_analysis['error'].notna()]['error'].unique()
            print(f"\n  計算錯誤:")
            for error in errors:
                print(f"    - {error}")
    else:
        print(f"\n  找到的买賣信號:")
        for i, pattern in enumerate(patterns[:10]):
            signal_type = "BUY" if pattern.signal == Signal.BUY else "SELL"
            print(f"    {i+1}. {pattern.timestamp} - {signal_type} (confidence: {pattern.confidence:.4f})")
        
        if len(patterns) > 10:
            print(f"    ... 及其他 {len(patterns) - 10} 區")

except Exception as e:
    print(f"  分析失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
if len(patterns) > 0:
    print("公式診斷: 正常")
    print(f"\n下一步: python run_optimization.py")
else:
    print("公式診斷: 橋欠")
    print("\n需要檢查:")
    print("  1. 技術指標計算是否正常")
    print("  2. 信號生成邏輯是否正確")
    print("  3. 參數是否需要調整")

print("="*70 + "\n")
