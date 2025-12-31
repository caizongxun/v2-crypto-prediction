# V2 優化 - 快速開始

## 🎯 3 分鐘總結

目標: 51.79% → 60-65%
特徵: 14 → 36 個
模型: RandomForest → LightGBM

## 🚀 執行命令

```bash
python train_models_v2.py
```

## 📈 預期結果

```
特徵形狀: (20524, 36)
方向準確率: ~62% ✅
vs v1: +10.21%
```

## 36 個特徵明細

### 原始 14 個 ✅
trend_score, volatility_score, direction_score, rsi, macd...

### 新增 22 個 ✨

**滯後特徵 (5)**
close_lag1, close_lag2, close_pct_lag1, direction_lag1, direction_lag2

**動量特徵 (4)**
rsi_momentum, rsi_acceleration, macd_momentum, bb_squeeze

**市場結構 (4)**
price_high_ratio, price_low_ratio, consecutive_up, consecutive_down

**相對波動率 (2)**
atr_ratio_to_mean, volume_ratio_to_mean

## 💡 模型輸出

### 看多信號
```json
{
  "direction": "BUY",
  "direction_probability": 0.65,
  "confidence": 0.30,
  "predicted_gain_pct": 0.2,
  "risk_reward_ratio": 1.33
}
```

### 決策規則
```python
if confidence > 0.25 and risk_reward_ratio > 1.0:
    do_trade()
```

## ⏱️ 時間成本

計算技術指標: ~10 秒
構建特徵: ~30 秒
訓練模型: ~3 分鐘
────────────────
總計: ~3 分鐘
