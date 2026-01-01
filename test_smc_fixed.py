#!/usr/bin/env python
"""
SMC 結構統計檢驗 - 修正的 zone 產生邏輯

說明:
此脚本不同於旧的實現，改正的主要邏輯是:

1. 只在真實的決悦點時生成 zone (供給 / 需求)
2. 去除每根 K 線都生成 zone 的失誤
3. 正確追蹤方向轉換，禁止重複生成
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加項目根目錄到 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from data import load_data
from indicators.smc import SmartMoneyStructure
from indicators.smc_visualizer import SMCVisualizer, create_smc_report


def test_smc_zone_generation():
    """檢驗 SMC zone 產生邏輯"""
    
    print("="*80)
    print("SMC 結構統計 - 修正的 Zone 產生邏輯")
    print("="*80)
    
    # 載入數據
    print("\n正在載入 BTC K 線數據...")
    df = load_data(
        start_date='2024-11-01',
        end_date='2024-12-31'
    )
    print(f✓ 載入完成: {len(df)} 筆數據")
    print(f繪載範圍: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 初始化 SMC 分析器
    print("\n正在計算 SMC 結構...")
    smc = SmartMoneyStructure(df, pivot_lookback=5, min_leg_length=3)
    smc.analyze()
    
    # 顯示結果
    print(f✓ 識別的腿部 (Legs): {len(smc.legs)} 個")
    print(f✓ 識別的樞紐點 (Pivots): {len(smc.pivots)} 個")
    print(f✓ 產生的 Zones: {len(smc.zones)} 個")
    
    # 詳稰的索接訊息
    supply_zones = [z for z in smc.zones if z.is_supply]
    demand_zones = [z for z in smc.zones if z.is_demand]
    
    print(f"  - Supply Zones: {len(supply_zones)} 個")
    print(f"  - Demand Zones: {len(demand_zones)} 個")
    
    # 顯示 Zone 位置継正
    if smc.zones:
        print("\n✨ Zone 位置継正 (只隊示前 5 個):")
        zones_df = smc.get_zones_df()
        print(zones_df.head(5).to_string(index=False))
        
        # 統計 zone 之間的間隔
        print("\n✨ Zone 間間隔統計:")
        if len(smc.zones) > 1:
            intervals = []
            for i in range(1, len(smc.zones)):
                interval = smc.zones[i].created_at_idx - smc.zones[i-1].created_at_idx
                intervals.append(interval)
            
            print(f"  - 平正間隔: {np.mean(intervals):.1f} K 線")
            print(f"  - 最短間間: {np.min(intervals)} K 線")
            print(f"  - 最長間間: {np.max(intervals)} K 線")
    
    # 创建可視化圖表
    print("\n✨ 正在生成可視化圖表...")
    visualizer = SMCVisualizer(figsize=(18, 10))
    
    # 標準檢驥 (最後 500 根 K 線)
    visualizer.plot(df, smc, start_idx=max(0, len(df) - 500))
    
    # 保存圖表
    output_dir = Path('./smc_reports')
    output_dir.mkdir(exist_ok=True)
    
    figure_path = output_dir / 'smc_zones_fixed.png'
    visualizer.save(str(figure_path))
    print(f✓ 圖表已保存: {figure_path}")
    
    # 生成報告
    print("\n✨ 正在生成報告...")
    report = create_smc_report(df, smc, str(output_dir))
    
    print("\n" + "="*80)
    print("✅ SMC 統計完成！")
    print("="*80)
    
    return df, smc, report


def analyze_zone_quality(smc: SmartMoneyStructure, df: pd.DataFrame) -> dict:
    """分析 zone 的品質指標"""
    
    print("\n✨ Zone 品質分析:")
    print("-" * 80)
    
    analysis = {
        'total_zones': len(smc.zones),
        'zone_sizes': [],
        'zone_spacing': []
    }
    
    # 分析每個 zone 的大小
    for i, zone in enumerate(smc.zones):
        size = zone.high - zone.low
        size_pct = (size / zone.mid) * 100
        analysis['zone_sizes'].append({
            'index': i,
            'type': 'Supply' if zone.is_supply else 'Demand',
            'size': size,
            'size_pct': size_pct,
            'created_at': zone.created_at_idx
        })
        print(f"Zone {i}: {zone.mid:.0f} | "
              f"Size: {size:.0f} ({size_pct:.2f}%) | "
              f"Type: {'Supply' if zone.is_supply else 'Demand'}")
    
    # 分析 zone 間間隔
    if len(smc.zones) > 1:
        print("\nZone 間間間統計:")
        for i in range(1, len(smc.zones)):
            spacing = smc.zones[i].created_at_idx - smc.zones[i-1].created_at_idx
            analysis['zone_spacing'].append(spacing)
            print(f"  Zone {i-1} → Zone {i}: {spacing} K 線")
    
    return analysis


if __name__ == '__main__':
    try:
        df, smc, report = test_smc_zone_generation()
        analysis = analyze_zone_quality(smc, df)
        
        # 檢驥: 应該比旧版本有明顯減少
        print("\n" + "="*80)
        print("✨ 驗証結果:")
        print("="*80)
        print(f"\n繪勘數據範圍: {len(df)} 筆 K 線")
        print(f创建的 Zones: {len(smc.zones)} 個")
        
        if len(smc.zones) == 0:
            print("\n⚠ 警告: 沒有出現 zone（可以調整參數）")
        elif len(smc.zones) > 100:
            print("\n⚠ 警告: Zone 數量過上，可能是邏輯有問題")
        else:
            print("\n✅ Zone 數量正常範圍 (預設: 20-50 個)")
        
        print("\n✅ 程式執行完成！")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
