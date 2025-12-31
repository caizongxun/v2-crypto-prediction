"""
模型輸出漄未漄橤漄淺漄算
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from data_handler import DataHandler
from indicators import IndicatorCalculator


def load_models():
    """載入訓練完成的模型"""
    with open('models/direction_model.pkl', 'rb') as f:
        direction_model = pickle.load(f)
    
    with open('models/gain_model.pkl', 'rb') as f:
        gain_model = pickle.load(f)
    
    with open('models/loss_model.pkl', 'rb') as f:
        loss_model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('formulas_results.json', 'r') as f:
        formulas_results = json.load(f)
    
    return direction_model, gain_model, loss_model, scaler, formulas_results


def apply_formulas(indicators, formulas_results):
    """應用三個公式"""
    
    # 變轴辞典粒式數処理
    def to_float(val):
        if hasattr(val, 'item'):
            return float(val.item())
        return float(val)
    
    # 變轴辞典戱管理
    def to_array(val):
        if isinstance(val, np.ndarray):
            return val
        return np.array(val)
    
    trend_params = formulas_results['trend_strength']['optimal_params']
    volatility_params = formulas_results['volatility_index']['optimal_params']
    direction_params = formulas_results['direction_confirmation']['optimal_params']
    
    # 輈度公式 1: 趨勤強度
    trend_strength = (
        to_float(trend_params['w_rsi']) * indicators['rsi'] +
        to_float(trend_params['w_macd']) * indicators['macd_line'] +
        to_float(trend_params['w_ema_diff']) * (indicators['ema_fast'] - indicators['ema_slow']) +
        to_float(trend_params['w_atr_ratio']) * (indicators['atr'] / (indicators['close'] + 1e-10)) +
        to_float(trend_params['w_volume']) * indicators['volume_ratio'] +
        to_float(trend_params['bias'])
    )
    
    # 輈度公式 2: 波動率指數
    volatility_index = (
        to_float(volatility_params['w_atr']) * (indicators['atr'] / (indicators['close'] + 1e-10)) +
        to_float(volatility_params['w_bollinger_width']) * (
            (indicators['upper_band'] - indicators['lower_band']) / (indicators['close'] + 1e-10)
        ) +
        to_float(volatility_params['w_volume_volatility']) * indicators['volume_volatility'] +
        to_float(volatility_params['w_price_change']) * indicators['price_change_rate'] +
        to_float(volatility_params['w_stochastic_range']) * (
            (indicators['k_line'] - indicators['d_line']) / 100
        ) +
        to_float(volatility_params['bias'])
    )
    
    # 輈度公式 3: 方向確認
    direction_confirmation = (
        to_float(direction_params['w_rsi_direction']) * (
            1 if indicators['rsi'][-1] > 50 else -1
        ) +
        to_float(direction_params['w_macd_direction']) * (
            1 if indicators['macd_line'][-1] > 0 else -1
        ) +
        to_float(direction_params['w_price_position']) * (
            (indicators['close'][-1] - indicators['sma_low'][-1]) / 
            (indicators['sma_high'][-1] - indicators['sma_low'][-1] + 1e-10)
        ) +
        to_float(direction_params['w_ema_slope']) * (
            1 if indicators['ema_fast'][-1] > indicators['ema_slow'][-1] else -1
        ) +
        to_float(direction_params['w_stochastic_direction']) * (
            1 if indicators['k_line'][-1] > indicators['d_line'][-1] else -1
        ) +
        to_float(direction_params['bias'])
    )
    
    return trend_strength, volatility_index, direction_confirmation


def predict(latest_ohlc, direction_model, gain_model, loss_model, scaler, formulas_results, indicators):
    """預測一根 K 線的信號數據"""
    
    # 應用公式
    trend_score, volatility_score, direction_score = apply_formulas(indicators, formulas_results)
    
    # 构建特征量
    features = np.array([
        trend_score,
        volatility_score,
        direction_score,
        indicators['rsi'][-1] / 100,
        indicators['macd_line'][-1],
        indicators['signal_line'][-1],
        (latest_ohlc['close'] - indicators['lower_band'][-1]) / 
            (indicators['upper_band'][-1] - indicators['lower_band'][-1] + 1e-10),
        indicators['atr'][-1] / (latest_ohlc['close'] + 1e-10),
        indicators['volume_ratio'][-1],
        indicators['k_line'][-1] / 100,
        indicators['d_line'][-1] / 100,
        abs((latest_ohlc['close'] - latest_ohlc['open']) / (latest_ohlc['close'] + 1e-10)),
        (latest_ohlc['high'] - latest_ohlc['low']) / (latest_ohlc['close'] + 1e-10),
        1 if indicators['ema_fast'][-1] > indicators['ema_slow'][-1] else 0
    ]).reshape(1, -1)
    
    # 標準化
    features_scaled = scaler.transform(features)
    
    # 預測
    direction_prob = direction_model.predict(features_scaled)[0]
    predicted_gain = gain_model.predict(features_scaled)[0]
    predicted_loss = loss_model.predict(features_scaled)[0]
    
    # 保護不會是負數
    predicted_gain = max(predicted_gain, 0.0001)
    predicted_loss = max(predicted_loss, 0.0001)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "ohlc": {
            "open": float(latest_ohlc['open']),
            "high": float(latest_ohlc['high']),
            "low": float(latest_ohlc['low']),
            "close": float(latest_ohlc['close']),
            "volume": float(latest_ohlc['volume'])
        },
        "formulas_scores": {
            "trend_strength": float(trend_score),
            "volatility_index": float(volatility_score),
            "direction_confirmation": float(direction_score)
        },
        "model_predictions": {
            "direction": "BUY" if direction_prob > 0.5 else "SELL",
            "direction_probability": float(direction_prob),
            "confidence": float(abs(direction_prob - 0.5) * 2),  # 0-1
            "predicted_gain_pct": float(predicted_gain * 100),
            "predicted_loss_pct": float(predicted_loss * 100),
            "risk_reward_ratio": float(predicted_gain / predicted_loss)
        },
        "technical_indicators": {
            "rsi": float(indicators['rsi'][-1]),
            "macd": float(indicators['macd_line'][-1]),
            "signal_line": float(indicators['signal_line'][-1]),
            "bollinger_bands": {
                "upper": float(indicators['upper_band'][-1]),
                "middle": float(indicators['middle_band'][-1]),
                "lower": float(indicators['lower_band'][-1])
            },
            "atr": float(indicators['atr'][-1]),
            "stochastic": {
                "k_line": float(indicators['k_line'][-1]),
                "d_line": float(indicators['d_line'][-1])
            },
            "ema": {
                "fast_12": float(indicators['ema_fast'][-1]),
                "slow_26": float(indicators['ema_slow'][-1]),
                "trend": "UP" if indicators['ema_fast'][-1] > indicators['ema_slow'][-1] else "DOWN"
            }
        }
    }


def main():
    print("\n" + "="*80)
    print("模型輸出漄未漄橤漄算")
    print("="*80)
    
    # 載入數據
    print("\n正在載入 BTC 數據...")
    handler = DataHandler()
    df = handler.load_data()
    
    # 載入公式
    print("正在載入公式結果...")
    direction_model, gain_model, loss_model, scaler, formulas_results = load_models()
    
    # 計算技術指標
    print("正在計算技術指標...")
    indicator_calc = IndicatorCalculator()
    indicators = indicator_calc.calculate_all(df)
    
    # 預測最後 5 根 K 線
    print("\n" + "="*80)
    print("最近 5 根 K 線的預測結果")
    print("="*80)
    
    for i in range(-5, 0):
        latest_ohlc = {
            'open': df.iloc[i]['open'],
            'high': df.iloc[i]['high'],
            'low': df.iloc[i]['low'],
            'close': df.iloc[i]['close'],
            'volume': df.iloc[i]['volume']
        }
        
        # 求單一根 K 線的信號
        output = predict(latest_ohlc, direction_model, gain_model, loss_model, scaler, formulas_results, indicators)
        
        print(f"\n第 {i+6} 根 K 線:")
        print(f"時間: {output['timestamp']}")
        print(f"價格: O={output['ohlc']['open']:.2f}, H={output['ohlc']['high']:.2f}, L={output['ohlc']['low']:.2f}, C={output['ohlc']['close']:.2f}")
        
        print(f"\n公式輸出:")
        print(f"  趨勤強度: {output['formulas_scores']['trend_strength']:.4f}")
        print(f"  波動率指數: {output['formulas_scores']['volatility_index']:.4f}")
        print(f"  方向確認: {output['formulas_scores']['direction_confirmation']:.4f}")
        
        print(f"\n模型預測:")
        print(f"  方向: {output['model_predictions']['direction']}")
        print(f"  方向樹佋室的椹: {output['model_predictions']['direction_probability']:.4f}")
        print(f"  置信度: {output['model_predictions']['confidence']:.2%}")
        print(f"  預測盈利: {output['model_predictions']['predicted_gain_pct']:.4f}%")
        print(f"  預測止損: {output['model_predictions']['predicted_loss_pct']:.4f}%")
        print(f"  風險報酬比: {output['model_predictions']['risk_reward_ratio']:.2f}")
        
        print(f"\n技術指標:")
        print(f"  RSI: {output['technical_indicators']['rsi']:.2f}")
        print(f"  MACD: {output['technical_indicators']['macd']:.6f}")
        print(f"  Signal: {output['technical_indicators']['signal_line']:.6f}")
        print(f"  Bollinger Bands: {output['technical_indicators']['bollinger_bands']['lower']:.2f} / "
              f"{output['technical_indicators']['bollinger_bands']['middle']:.2f} / "
              f"{output['technical_indicators']['bollinger_bands']['upper']:.2f}")
        print(f"  ATR: {output['technical_indicators']['atr']:.2f}")
        print(f"  Stochastic: K={output['technical_indicators']['stochastic']['k_line']:.2f}, "
              f"D={output['technical_indicators']['stochastic']['d_line']:.2f}")
        print(f"  EMA: Fast={output['technical_indicators']['ema']['fast_12']:.2f}, "
              f"Slow={output['technical_indicators']['ema']['slow_26']:.2f}, "
              f"Trend={output['technical_indicators']['ema']['trend']}")
        
        print("-" * 80)
    
    # 保存後期結果
    print(f"\n全量輸出 JSON 已保存至: model_output.json")
    with open('model_output.json', 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
