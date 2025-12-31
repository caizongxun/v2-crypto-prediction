#!/usr/bin/env python3
"""
粗訪兼脫韓目標的優化執行器

步驟:
1. python download_and_save_data.py  (第一次執行，下載並保存數據)
2. python run_optimization.py         (快速版 - 10 分鐘)
3. 查看 results/ 文件夾的 JSON 結果
4. 使用最優參數進行回測
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("黃金公式優化器 - 粗訪版本")
print("="*70)

try:
    print("\n[1/5] 檢查依賴...")
    from config import HF_TOKEN
    from formulas.golden_formula_v2 import GoldenFormulaV2, Signal
    from formulas.golden_formula_v2_config import GoldenFormulaV2Config
    from backtest.backtest_engine import BacktestEngine
    from optimization.parameter_optimizer import ParameterOptimizer
    print("   所有依賴已加載")
except ImportError as e:
    print(f"   依賴不足: {e}")
    print("\n   解決步驟:")
    print("   pip install pandas numpy huggingface_hub optuna scikit-learn pyarrow")
    sys.exit(1)

try:
    print("\n[2/5] 驗證 Hugging Face Token...")
    if not HF_TOKEN or HF_TOKEN == "your_token_here":
        print("   HF_TOKEN 未設定")
        print("\n   解決步驟:")
        print("   1. 從 https://huggingface.co/settings/tokens 獲取 token")
        print("   2. 編輯 config.py, 設置 HF_TOKEN")
        print("   3. 重新運行腳本")
        sys.exit(1)
    print("   Token 已設定")
except Exception as e:
    print(f"   例外: {e}")
    sys.exit(1)

try:
    print("\n[3/5] 加載 BTC 15m 數據 (2024-01-01 ~ 2024-12-31)...")
    
    # 本地快取路徑
    local_data_path = "./data/btc_15m.parquet"
    
    # 檢查本地文件是否存在
    if not os.path.exists(local_data_path):
        print(f"   本地文件不存在: {local_data_path}")
        print("\n   解決步驟:")
        print("   1. 先運行: python download_and_save_data.py")
        print("   2. 然後運行: python run_optimization.py")
        sys.exit(1)
    
    print(f"   使用本地快取: {local_data_path}")
    
    # 加載本地數據
    df = pd.read_parquet(local_data_path)
    
    # 驗證數據
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        print(f"   錯誤: 缺少列: {[c for c in required_columns if c not in df.columns]}")
        sys.exit(1)
    
    # 過濾日期範圍 (2024-01-01 ~ 2024-12-31)
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-12-31 23:59:59')
    
    original_len = len(df)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    print(f"   過濾後數據: {len(df)} 根 K 線 (原始: {original_len} 根)")
    print(f"   時間範圍: {df.index[0]} ~ {df.index[-1]}")
    
    if len(df) == 0:
        print("   錯誤: 沒有符合日期範圍的數據")
        sys.exit(1)
    
except Exception as e:
    print(f"   數據加載失敗: {e}")
    print("\n   解決步驟:")
    print("   1. 先運行: python download_and_save_data.py")
    print("   2. 確認本地文件存在: ./data/btc_15m.parquet")
    print("   3. 然後運行: python run_optimization.py")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[4/5] 定義優化目標函數...")
    
    class SimpleOptimizer:
        """粗訪優化器"""
        def __init__(self, df):
            self.df = df
            self.engine = BacktestEngine(initial_capital=10000)
        
        def objective(self, params):
            """目標函數 (Sharpe Ratio)"""
            try:
                from formulas.golden_formula_v2_config import (
                    TrendConfig, MomentumConfig, VolumeConfig
                )
                
                trend_cfg = TrendConfig(
                    fast_ema_period=int(params['fast_ema']),
                    slow_ema_period=int(params['slow_ema']),
                    supertrend_period=int(params['supertrend_period']),
                    supertrend_multiplier=params['supertrend_multiplier'],
                    adx_min_threshold=params['adx_threshold']
                )
                momentum_cfg = MomentumConfig(
                    rsi_period=int(params['rsi_period']),
                    rsi_oversold=params['rsi_oversold'],
                    rsi_overbought=params['rsi_overbought'],
                    roc_period=int(params['roc_period'])
                )
                volume_cfg = VolumeConfig(
                    volume_spike_multiplier=params['volume_spike'],
                    vwap_deviation_percent=params['vwap_deviation']
                )
                
                config = GoldenFormulaV2Config(
                    trend_config=trend_cfg,
                    momentum_config=momentum_cfg,
                    volume_config=volume_cfg
                )
                
                config.entry_config.trend_weight = params['trend_weight']
                config.entry_config.momentum_weight = params['momentum_weight']
                config.entry_config.volume_weight = params['volume_weight']
                config.entry_config.min_confidence_threshold = params['confidence_threshold']
                
                formula = GoldenFormulaV2(config)
                patterns, _ = formula.analyze(self.df)
                
                if len(patterns) < 5:
                    return -np.inf
                
                signals = [(p.index, 1 if p.signal == Signal.BUY else -1) for p in patterns]
                result = self.engine.run(self.df, signals)
                
                return result.sharpe_ratio if not np.isnan(result.sharpe_ratio) else -np.inf
            except Exception as e:
                return -np.inf
    
    optimizer = SimpleOptimizer(df)
    print("   優化器已初始化")
except Exception as e:
    print(f"   優化器初始化失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[5/5] 執行優化... (預計 10 分鐘)\n")
    
    os.makedirs("results", exist_ok=True)
    
    # 參數樣本外優化空間 (推薦範圍)
    param_space = {
        'fast_ema': (10, 20),
        'slow_ema': (40, 80),
        'supertrend_period': (8, 15),
        'supertrend_multiplier': (2.0, 4.0),
        'adx_threshold': (20, 30),
        'rsi_period': (12, 16),
        'rsi_oversold': (25, 35),
        'rsi_overbought': (65, 75),
        'roc_period': (10, 15),
        'volume_spike': (1.3, 2.0),
        'vwap_deviation': (0.8, 1.5),
        'trend_weight': (0.35, 0.45),
        'momentum_weight': (0.25, 0.35),
        'volume_weight': (0.15, 0.25),
        'confidence_threshold': (0.60, 0.70)
    }
    
    param_opt = ParameterOptimizer(optimizer.objective)
    
    # 執行 Random Search (更快)
    print("正在執行 Random Search (100 次試驗)...\n")
    start_time = time.time()
    
    result = param_opt.random_search(
        param_space=param_space,
        n_trials=100,
        random_state=42,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # 打印結果
    print("\n\n" + "="*70)
    print("優化結果")
    print("="*70)
    print(f"\n耗時: {elapsed:.2f} 秒 ({elapsed/60:.1f} 分鐘)")
    print(f"試驗次數: {result.total_trials}")
    print(f"\n最佳得分 (Sharpe Ratio): {result.best_score:.4f}")
    print(f"\n最佳參數:")
    for key, value in sorted(result.best_params.items()):
        if isinstance(value, float):
            print(f"  {key:25s} = {value:.4f}")
        else:
            print(f"  {key:25s} = {value}")
    
    # 保存結果
    ParameterOptimizer.export_results(result, "results/optimization_result.json")
    
    print(f"\n結果已保存到: results/optimization_result.json")
    print("\n" + "="*70)
    print("下一步:")
    print("  1. 查看 JSON 文件中的 best_params")
    print("  2. 使用最優參數進行回測驗證")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n優化失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
