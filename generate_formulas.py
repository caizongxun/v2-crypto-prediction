"""
含比外慢軟的 AI 逆向推理公式生成器

使用遺傳演算法自動發現 3 個黃金公式
"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_sma,
    calculate_ema,
    calculate_volume_sma,
    calculate_stochastic
)
from formulas import FormulaGenerator
import json
from pathlib import Path

# 加載環境變量
load_dotenv()


def load_data():
    """
    從 HuggingFace 加載 BTC K 線數據
    """
    from data import load_btc_data
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("缺少 HF_TOKEN 環境變量")
    
    df = load_btc_data(
        hf_token=hf_token,
        start_date='2024-06-01',
        end_date='2024-12-31'
    )
    
    if df is None:
        raise ValueError("無法加載數據")
    
    return df


def prepare_indicators(df: pd.DataFrame) -> dict:
    """
    計算所有技術指標
    """
    print("\n正在計算技術指標...")
    
    # MACD
    macd_line, signal_line, macd_histogram = calculate_macd(
        df['close'], fast=12, slow=26, signal=9
    )
    
    # Bollinger Bands
    middle_band, upper_band, lower_band = calculate_bollinger_bands(
        df['close'], period=20, num_std=2.0
    )
    
    # Stochastic
    k_line, d_line = calculate_stochastic(
        df['high'], df['low'], df['close'], 
        period=14, smooth_k=3, smooth_d=3
    )
    
    indicators = {
        # 基本 OHLCV
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume'],
        
        # RSI
        'rsi': calculate_rsi(df['close'], period=14),
        
        # MACD
        'macd_line': macd_line,
        'signal_line': signal_line,
        'macd_histogram': macd_histogram,
        
        # Bollinger Bands
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band,
        
        # ATR
        'atr': calculate_atr(df['high'], df['low'], df['close'], period=14),
        
        # 移動平均
        'sma_20': calculate_sma(df['close'], period=20),
        'ema_fast': calculate_ema(df['close'], period=12),
        'ema_slow': calculate_ema(df['close'], period=26),
        
        # 成交量
        'volume_sma': calculate_volume_sma(df['volume'], period=20),
        'volume_ratio': df['volume'] / (calculate_volume_sma(df['volume'], period=20) + 1e-10),
        
        # Stochastic
        'k_line': k_line,
        'd_line': d_line,
    }
    
    print(f"  計算了 {len(indicators)} 個指標")
    return indicators


def main():
    print("="*70)
    print("含比外慢軟的 AI 逆向推理公式生成器")
    print("="*70)
    
    # 載入費給基本數據
    print("\n正在載入 BTC 15分鐘 K線數據...")
    df = load_data()
    print(f"  成功載入 {len(df)} 筆 K線數據")
    print(f"  時間範圍: {df.index[0]} ~ {df.index[-1]}")
    
    # 計算技術指標
    indicators = prepare_indicators(df)
    
    # 創建公式生成器
    print("\n正在初始化公式生成器...")
    generator = FormulaGenerator(
        df=df,
        indicators_dict=indicators,
        population_size=50,
        generations=20,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # 發現公式
    formulas = generator.generate()
    
    # 保存結果
    print("\n" + "="*70)
    print("儲存公式結果...")
    print("="*70)
    
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_range': f"{df.index[0]} ~ {df.index[-1]}",
        'total_candles': len(df),
        'formulas': {}
    }
    
    for formula_type, formula_data in formulas.items():
        coefficients = formula_data['coefficients']
        fitness = formula_data['fitness']
        
        results['formulas'][formula_type] = {
            'coefficients': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in coefficients.items()},
            'fitness': float(fitness),
            'description': get_formula_description(formula_type)
        }
        
        print(f"\n{formula_type}:")
        print(f"  最優係數: {coefficients}")
        print(f"  適應度: {fitness:.4f}")
    
    # 保存到 JSON
    output_path = Path('formulas_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果已保存至: {output_path}")
    
    return formulas


def get_formula_description(formula_type: str) -> str:
    """
    取得公式描述
    """
    descriptions = {
        'trend_strength': '趨勢強度 - 輸出 0-1，數值漸正趨勢強度',
        'volatility_index': '波動率 - 輸出 0-1，數值漸正波動幅度',
        'direction_confirmation': '方向確認 - 輸出 0-1，>0.5看多 <0.5看空'
    }
    return descriptions.get(formula_type, '')


if __name__ == '__main__':
    try:
        formulas = main()
        
        print("\n" + "="*70)
        print("完成! 發現了 3 個黃金公式")
        print("="*70)
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
