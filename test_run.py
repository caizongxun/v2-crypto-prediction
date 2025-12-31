#!/usr/bin/env python3
"""
測試脚本 - 驗證環境和數據加載
"""

import os
import sys
from config import HF_TOKEN
from data import DataLoader
from formulas import GoldenFormulaV1

def test_environment():
    """測試環境配置"""
    print("=" * 70)
    print("環境検查")
    print("=" * 70)
    
    if not HF_TOKEN:
    鄦錯誤: HF_TOKEN 未設定")
        print("請在 .env 檔案中設定 HF_TOKEN")
        return False
    
    print("HF_TOKEN: 已設定")
    print("Python 版本:", sys.version)
    print()
    return True

def test_data_loader():
    """測試數據加載器"""
    print("=" * 70)
    print("測試數據加載器")
    print("=" * 70)
    
    try:
        loader = DataLoader(hf_token=HF_TOKEN)
        
        print("列出可用交易對...")
        pairs = loader.list_available_pairs()
        
        if pairs:
            print(f"找到 {len(pairs)} 個交易對:")
            for pair in pairs[:10]:
                print(f"  - {pair}")
            
            if len(pairs) > 10:
                print(f"  ... 還有 {len(pairs) - 10} 個")
        else:
            print("未找到任何交易對")
            return False
        
        print()
        return True
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_golden_formula():
    """測試黃金公式"""
    print("=" * 70)
    print("測試黃金公式 V1")
    print("=" * 70)
    
    try:
        import pandas as pd
        import numpy as np
        
        formula = GoldenFormulaV1(lookback_period=20)
        
        # 建立模擬數據
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        np.random.seed(42)
        
        data = {
            'open_time': dates,
            'open': 40000 + np.cumsum(np.random.randn(100) * 100),
            'high': 40100 + np.cumsum(np.random.randn(100) * 100),
            'low': 39900 + np.cumsum(np.random.randn(100) * 100),
            'close': 40000 + np.cumsum(np.random.randn(100) * 100),
            'volume': np.random.randint(100, 1000, 100),
        }
        
        df = pd.DataFrame(data)
        
        print("检測樣式...")
        patterns = formula.detect_interval_reversal(df, min_pattern_strength=0.7)
        
        print(f"找到 {len(patterns)} 個樣式")
        
        if patterns:
            print("\n最近 5 個樣式:")
            for pattern in patterns[-5:]:
                print(f"  {pattern.timestamp}: {pattern.signal.value} (信忆: {pattern.confidence:.2f})")
        
        summary = formula.get_patterns_summary()
        print(f"\n摆要:")
        print(f"  總計: {summary['total']}")
        print(f"  買入: {summary['buy_signals']}")
        print(f"  賣出: {summary['sell_signals']}")
        print(f"  平均信忆: {summary['average_confidence']:.2f}")
        
        print()
        return True
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("\n")
    print("*" * 70)
    print("V2 Crypto Prediction System - 環境測試")
    print("*" * 70)
    print()
    
    results = []
    
    # 測試環境
    results.append(("環境検查", test_environment()))
    
    # 測試數據加載器
    results.append(("數據加載器", test_data_loader()))
    
    # 測試黃金公式
    results.append(("黃金公式 V1", test_golden_formula()))
    
    # 總結
    print("=" * 70)
    print("測試結果摆要")
    print("=" * 70)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20s}: {status}")
    
    print("=" * 70)
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n所有測試通過！環境配置正確。")
        return 0
    else:
        print("\n某些測試失敗。請検查上面的錯誤信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
