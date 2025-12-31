# 模型輸出結構詳解

## 📤 完整輸出示例

```json
{
  "timestamp": "2025-12-31T10:14:00.000000",
  
  "ohlc": {
    "open": 42150.45,
    "high": 42580.80,
    "low": 42100.30,
    "close": 42450.65,
    "volume": 1240.5
  },
  
  "formulas_scores": {
    "trend_strength": 0.3847,
    "volatility_index": 0.2156,
    "direction_confirmation": 0.5621
  },
  
  "model_predictions": {
    "direction": "BUY",
    "direction_probability": 0.6234,
    "confidence": 0.2468,
    "predicted_gain_pct": 0.1623,
    "predicted_loss_pct": 0.1756,
    "risk_reward_ratio": 0.9241
  },
  
  "technical_indicators": {
    "rsi": 58.34,
    "macd": 0.00456,
    "signal_line": 0.00382,
    "bollinger_bands": {
      "upper": 43200.50,
      "middle": 42400.00,
      "lower": 41600.50
    },
    "atr": 285.34,
    "stochastic": {
      "k_line": 62.45,
      "d_line": 58.92
    },
    "ema": {
      "fast_12": 42350.45,
      "slow_26": 42200.30,
      "trend": "UP"
    }
  }
}
```

## 📊 輸出欄位完整解讀

### OHLC 數據 (K 線信息)
- **open**: 開盤價
- **high**: 最高價
- **low**: 最低價
- **close**: 收盤價
- **volume**: 成交量

### 公式輸出 (三個核心信號)

**趨勢強度** (Trend Strength)
- 範圍: -1 到 +1
- 正值 = 上升趨勢，負值 = 下降趨勢
- 絕對值越大，趨勢越強

**波動率指數** (Volatility Index)
- 範圍: -1 到 +1
- 絕對值表示波動幅度
- 高值 = 市場波動大，低值 = 盤整

**方向確認** (Direction Confirmation)
- 範圍: -1 到 +1
- 正值 = 看多信號，負值 = 看空信號
- 絕對值表示確認強度

### 模型預測 (交易決策)

**Direction**: BUY 或 SELL
- 基於 direction_probability > 0.5 判斷

**Direction Probability**: 0-1 之間
- 下一根 K 線上升的概率
- 0.5 = 50% (完全不知道)
- 0.7 = 70% (傾向看多)

**Confidence**: 0-1 之間
- 公式: abs(probability - 0.5) * 2
- 衡量模型對該預測的確定程度

**Predicted Gain %**: 盈利幅度
- 下一根 K 線最高價相對當前價的漲幅
- 0.16% 表示預計上升 0.16%

**Predicted Loss %**: 止損幅度
- 下一根 K 線最低價相對當前價的跌幅
- 0.17% 表示預計下跌 0.17%

**Risk Reward Ratio**: 風險報酬比
- gain / loss
- 1.0 = 1:1 (風險與報酬相等)
- 2.0 = 2:1 (盈利比虧損 2 倍，優秀)
- 0.5 = 0.5:1 (虧損比盈利 2 倍，不值得)

### 技術指標 (診斷信息)

- **RSI**: 相對強弱指數 (0-100)
- **MACD/Signal**: 動量指標
- **Bollinger Bands**: 價格波動帶
- **ATR**: 真實波幅
- **Stochastic**: 隨機指標
- **EMA**: 指數移動平均線
