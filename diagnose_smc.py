#!/usr/bin/env python
"""
SMC Order Block 模式診斷
使用正常的 detect_order_blocks 邏輯來看看为什麼 OB 會過幾
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data import load_data
from model_ensemble_gui import SmartMoneyStructure
import pandas as pd

def diagnose():
    print("="*80)
    print("SMC Order Block Diagnostic")
    print("="*80)
    
    # Load data
    df = load_data(start_date='2024-11-01', end_date='2024-12-31')
    print(f"\n[DATA] Loaded {len(df)} bars")
    
    # Analyze
    smc = SmartMoneyStructure(swing_length=50)
    result = smc.analyze(df)
    
    pivots = result['pivots']
    structures = result['structures']
    order_blocks = result['order_blocks']
    
    # Statistics
    high_pivots = len(pivots['high'])
    low_pivots = len(pivots['low'])
    total_pivots = high_pivots + low_pivots
    
    print(f"\n[PIVOTS]")
    print(f"  High Pivots: {high_pivots}")
    print(f"  Low Pivots: {low_pivots}")
    print(f"  Total: {total_pivots}")
    
    print(f"\n[STRUCTURES]")
    print(f"  Total: {len(structures)}")
    bos_count = len([s for s in structures if s['type'] == 'BOS'])
    choch_count = len([s for s in structures if s['type'] == 'CHoCH'])
    print(f"  BOS: {bos_count}")
    print(f"  CHoCH: {choch_count}")
    
    print(f"\n[ORDER BLOCKS]")
    print(f"  Total: {len(order_blocks)}")
    
    if order_blocks:
        bearish_ob = len([ob for ob in order_blocks if ob['type'] == 'bearish'])
        bullish_ob = len([ob for ob in order_blocks if ob['type'] == 'bullish'])
        print(f"  Bearish OB: {bearish_ob}")
        print(f"  Bullish OB: {bullish_ob}")
        
        mitigated = len([ob for ob in order_blocks if ob['is_mitigated']])
        print(f"  Mitigated: {mitigated}")
        
        # Size statistics
        print(f"\n[OB SIZE STATISTICS]")
        sizes = []
        for ob in order_blocks:
            size = ob['high'] - ob['low']
            size_pct = (size / ob['high'] * 100) if ob['high'] > 0 else 0
            sizes.append(size_pct)
            print(f"  OB at idx {ob['start_idx']:5d}-{ob['end_idx']:5d}: "
                  f"Price range {ob['low']:.0f}-{ob['high']:.0f}, "
                  f"Size: {size_pct:.2f}%, "
                  f"Type: {ob['type']}, "
                  f"Mitigated: {ob['is_mitigated']}")
        
        if sizes:
            import numpy as np
            print(f"\n  Avg size: {np.mean(sizes):.2f}%")
            print(f"  Min size: {np.min(sizes):.2f}%")
            print(f"  Max size: {np.max(sizes):.2f}%")
    
    # Ratio analysis
    print(f"\n[RATIO ANALYSIS]")
    if total_pivots > 0:
        print(f"  OB / Pivots: {len(order_blocks) / total_pivots:.2f}")
    print(f"  Data points per OB: {len(df) / max(1, len(order_blocks)):.0f}")
    
    print("\n" + "="*80)
    print("\n[EXPECTED]")
    print("For 2 months (approx 5000+ bars) with 15m candles:")
    print("  - Pivots: 50-100 per month (more frequent for SMC)")
    print("  - OB: Should be 5-15 total (not one every 10-20 bars)")
    print("  - Each OB should be 5-50 bars wide, not entire trend")
    
if __name__ == '__main__':
    try:
        diagnose()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
