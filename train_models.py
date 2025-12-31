"""
第二階段: 機器學習訓練

使用公式輸出訓練模組，預測開單點位
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from dotenv import load_dotenv

from data import load_btc_data
from indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_ema,
    calculate_volume_sma,
    calculate_stochastic
)
from formulas import (
    TrendStrengthFormula,
    VolatilityIndexFormula,
    DirectionConfirmationFormula
)
from models import ModelTrainer

# 加載環境變量
load_dotenv()


def load_data():
    """
    從 HuggingFace 加載 BTC K 線數據
    """
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


def calculate_indicators(df: pd.DataFrame) -> dict:
    """
    計算技術指標
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
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume'],
        'rsi': calculate_rsi(df['close'], period=14),
        'macd_line': macd_line,
        'signal_line': signal_line,
        'macd_histogram': macd_histogram,
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'atr': calculate_atr(df['high'], df['low'], df['close'], period=14),
        'sma_20': calculate_sma(df['close'], period=20),
        'ema_fast': calculate_ema(df['close'], period=12),
        'ema_slow': calculate_ema(df['close'], period=26),
        'volume_sma': calculate_volume_sma(df['volume'], period=20),
        'volume_ratio': df['volume'] / (calculate_volume_sma(df['volume'], period=20) + 1e-10),
        'k_line': k_line,
        'd_line': d_line,
    }
    
    print(f"  計算了 {len(indicators)} 個指標")
    return indicators


def calculate_sma(close: pd.Series, period: int = 20) -> pd.Series:
    return close.rolling(window=period).mean()


def calculate_ema(close: pd.Series, period: int = 20) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def load_formulas() -> dict:
    """
    加載一先前生成的公式結果
    """
    results_path = Path('formulas_results.json')
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} 不存在")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_formulas(
    indicators: dict,
    formulas_results: dict
) -> dict:
    """
    應用三個公式計算輸出
    """
    print("\n正在應用公式...")
    
    # 取出係數
    trend_coef = formulas_results['formulas']['trend_strength']['coefficients']
    volatility_coef = formulas_results['formulas']['volatility_index']['coefficients']
    direction_coef = formulas_results['formulas']['direction_confirmation']['coefficients']
    
    # 計算公式輸出
    trend_formula = TrendStrengthFormula()
    trend_score = trend_formula.calculate(indicators, trend_coef)
    
    volatility_formula = VolatilityIndexFormula()
    volatility_score = volatility_formula.calculate(indicators, volatility_coef)
    
    direction_formula = DirectionConfirmationFormula()
    direction_score = direction_formula.calculate(indicators, direction_coef)
    
    formulas_outputs = {
        'trend_strength': trend_score,
        'volatility_index': volatility_score,
        'direction_confirmation': direction_score
    }
    
    print(f"  trend_strength: {trend_score.mean():.4f} ± {trend_score.std():.4f}")
    print(f"  volatility_index: {volatility_score.mean():.4f} ± {volatility_score.std():.4f}")
    print(f"  direction_confirmation: {direction_score.mean():.4f} ± {direction_score.std():.4f}")
    
    return formulas_outputs


def main():
    print("="*70)
    print("第二階段: 機器學習訓練")
    print("="*70)
    
    # 1. 載入數據
    print("\n正在載入 BTC K線數據...")
    df = load_data()
    print(f"  加載 {len(df)} 筆數據")
    
    # 2. 計算技術指標
    indicators = calculate_indicators(df)
    
    # 3. 加載並應用公式
    formulas_results = load_formulas()
    print(f"  已加載公式結果")
    
    formulas_outputs = apply_formulas(indicators, formulas_results)
    
    # 4. 訓練模組
    print("\n" + "="*70)
    print("正在訓練模組...")
    print("="*70)
    
    trainer = ModelTrainer(df, formulas_results)
    
    # 準備特彉
    X, y = trainer.prepare_features(indicators, formulas_outputs)
    
    # 訓練
    results = trainer.train(X, y, test_size=0.2, val_size=0.1)
    
    # 保存模型
    trainer.save_models(output_dir='models')
    
    print("\n" + "="*70)
    print("完成! 模型訓練完成")
    print("="*70)
    
    return trainer, results


if __name__ == '__main__':
    try:
        trainer, results = main()
        print("\n訓練結果:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
