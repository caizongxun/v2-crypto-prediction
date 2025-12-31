# 虛擬貨幣預測系統 - 完整公式參考指南

## 目錄
1. [趨勢指標公式](#趨勢指標)
2. [方向指標公式](#方向指標)
3. [波動性指標公式](#波動性指標)
4. [集合信號公式](#集合信號)
5. [績效評估公式](#績效評估)
6. [代碼位置地圖](#代碼位置地圖)

---

## 趨勢指標

### 基礎趨勢指標 (indicators_implementation.py)

#### 公式 1: 指數移動平均 (EMA)
```
fast_ema = EMA(close, 23)
slow_ema = EMA(close, 90)

EMA 計算公式:
EMA(t) = close(t) × 2/(n+1) + EMA(t-1) × (1 - 2/(n+1))

其中:
- close(t) = 第 t 根 K 線的收盤價
- n = EMA 周期
- EMA(t-1) = 前一個 EMA 值
```

**代碼位置:** `indicators_implementation.py` 第 52-53 行
```python
fast_ema = close.ewm(span=TrendIndicator.FAST_EMA_PERIOD, adjust=False).mean()
slow_ema = close.ewm(span=TrendIndicator.SLOW_EMA_PERIOD, adjust=False).mean()
```

#### 公式 2: 真實波幅 (True Range)
```
TR = max(H - L, |H - C(t-1)|, |L - C(t-1)|)

其中:
- H = 最高價
- L = 最低價
- C(t-1) = 上一根 K 線的收盤價
```

**代碼位置:** `indicators_implementation.py` 第 55-59 行
```python
high_low = high - low
high_close = np.abs(high - close.shift())
low_close = np.abs(low - close.shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
```

#### 公式 3: 平均真實波幅 (ATR)
```
ATR = EMA(TR, 10)

其中:
- TR = 真實波幅
- 10 = ATR 周期
```

**代碼位置:** `indicators_implementation.py` 第 61 行
```python
atr = tr.ewm(span=TrendIndicator.ATR_PERIOD, adjust=False).mean()
```

#### 公式 4: EMA 比率
```
EMA_ratio = (fast_ema - slow_ema) / slow_ema × 100

說明:
- 快速 EMA 減去緩慢 EMA 的百分比差異
- 正值表示上升趨勢
- 負值表示下降趨勢
```

**代碼位置:** `indicators_implementation.py` 第 64 行
```python
ema_ratio = (fast_ema - slow_ema) / slow_ema * 100
```

#### 公式 5: ATR 比率
```
ATR_ratio = (ATR / close) × 100

說明:
- 波動幅度相對於收盤價的百分比
- 用於衡量市場波動程度
```

**代碼位置:** `indicators_implementation.py` 第 68 行
```python
atr_ratio = (atr / close) * 100
```

#### 公式 6: 趨勢分數
```
trend_score = tanh(ema_ratio / 2) × (1 - exp(-atr_ratio / 0.5))

說明:
- tanh(x) 將 EMA 比率規範化至 [-1, 1]
- (1 - exp(-x)) 將 ATR 比率轉換為 [0, 1] 的權重
- 兩者相乘得到最終趨勢分數 [-1, 1]
```

**代碼位置:** `indicators_implementation.py` 第 71 行
```python
trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-atr_ratio / 0.5))
```

#### 公式 7: 規範化趨勢值
```
trend_value = (trend_score + 1) / 2

其中:
- trend_score ∈ [-1, 1]
- trend_value ∈ [0, 1]

解釋:
- 0.6+ = 強上升趨勢 (買入信號)
- 0.5-0.6 = 上升趨勢
- 0.4-0.5 = 中立
- 0.3-0.4 = 下降趨勢
- 0-0.3 = 強下降趨勢 (賣出信號)
```

**代碼位置:** `indicators_implementation.py` 第 74 行
```python
trend_value = (trend_score + 1) / 2
```

---

## 方向指標

### 基礎方向指標 (indicators_implementation.py)

#### 公式 8: 相對強弱指數 (RSI)

**步驟 1: 計算價格變化**
```
delta = close(t) - close(t-1)
gain = max(delta, 0)      # 只保留正數
loss = max(-delta, 0)     # 只保留負數的絕對值
```

**代碼位置:** `indicators_implementation.py` 第 103-108 行
```python
delta = close.diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
```

**步驟 2: 計算平均收益和平均虧損**
```
avg_gain = EMA(gain, 10)
avg_loss = EMA(loss, 10)
```

**代碼位置:** `indicators_implementation.py` 第 110-111 行
```python
avg_gain = pd.Series(gain).ewm(span=DirectionIndicator.RSI_PERIOD, adjust=False).mean()
avg_loss = pd.Series(loss).ewm(span=DirectionIndicator.RSI_PERIOD, adjust=False).mean()
```

**步驟 3: 計算 RSI**
```
RS = avg_gain / avg_loss
RSI = 100 - (100 / (1 + RS))

值域: [0, 100]
- > 70: 超買
- < 30: 超賣
```

**代碼位置:** `indicators_implementation.py` 第 114-115 行
```python
rs = avg_gain / (avg_loss + 1e-10)
rsi = 100 - (100 / (1 + rs))
```

#### 公式 9: 變動率 (ROC)
```
ROC = ((close(t) - close(t-8)) / close(t-8)) × 100

說明:
- 測量價格在 8 個周期內的百分比變化
- 正值表示上升
- 負值表示下降
```

**代碼位置:** `indicators_implementation.py` 第 118 行
```python
roc = ((close - close.shift(DirectionIndicator.ROC_PERIOD)) / close.shift(DirectionIndicator.ROC_PERIOD)) * 100
```

#### 公式 10: RSI 信號規範化
```
rsi_signal = (RSI - 50) / 50

其中:
- RSI ∈ [0, 100]
- rsi_signal ∈ [-1, 1]
```

**代碼位置:** `indicators_implementation.py` 第 121 行
```python
rsi_signal = (rsi - 50) / 50
```

#### 公式 11: ROC 信號規範化
```
roc_signal = tanh(ROC / 5)

說明:
- 將 ROC 轉換到 [-1, 1] 範圍
- tanh 函數平滑異常值
```

**代碼位置:** `indicators_implementation.py` 第 125 行
```python
roc_signal = np.tanh(roc / 5)
```

#### 公式 12: 方向分數
```
direction_score = (rsi_signal + roc_signal) / 2

說明:
- 結合 RSI 和 ROC 的信號
- 值域: [-1, 1]
```

**代碼位置:** `indicators_implementation.py` 第 128 行
```python
direction_score = (rsi_signal + roc_signal) / 2
```

#### 公式 13: 規範化方向值
```
direction_value = (direction_score + 1) / 2

其中:
- direction_score ∈ [-1, 1]
- direction_value ∈ [0, 1]
```

**代碼位置:** `indicators_implementation.py` 第 131 行
```python
direction_value = (direction_score + 1) / 2
```

---

## 波動性指標

### 基礎波動性指標 (indicators_implementation.py)

#### 公式 14: 簡單移動平均 (SMA)
```
SMA = Σ(close(t-i) for i=0 to n-1) / n

其中:
- n = 39 (SMA 周期)
```

**代碼位置:** `indicators_implementation.py` 第 163 行
```python
sma = close.rolling(window=VolatilityIndicator.SMA_PERIOD).mean()
```

#### 公式 15: 標準差 (STDEV)
```
STDEV = sqrt(Σ((close(i) - SMA)^2) / n)

其中:
- n = 39 (周期)
- SMA = 簡單移動平均
```

**代碼位置:** `indicators_implementation.py` 第 166 行
```python
std = close.rolling(window=VolatilityIndicator.SMA_PERIOD).std()
```

#### 公式 16: 布林通道 (Bollinger Bands)
```
upper = SMA + (2.6 × STDEV)
lower = SMA - (2.6 × STDEV)

說明:
- 2.6 是標準差倍數
- 約 98.8% 的數據在此範圍內
```

**代碼位置:** `indicators_implementation.py` 第 169-170 行
```python
upper = sma + (VolatilityIndicator.BB_STD_MULTIPLIER * std)
lower = sma - (VolatilityIndicator.BB_STD_MULTIPLIER * std)
```

#### 公式 17: 波動性比率
```
volatility_ratio = (STDEV / SMA) × 100

說明:
- > 2%: 高波動性
- < 1%: 低波動性
```

**代碼位置:** `indicators_implementation.py` 第 174 行
```python
volatility_ratio = (std / sma) * 100
```

#### 公式 18: 波動性分數
```
volatility_score = sqrt(volatility_ratio / 2)

說明:
- 使用平方根函數平滑數據
```

**代碼位置:** `indicators_implementation.py` 第 177 行
```python
volatility_score = np.sqrt(volatility_ratio / 2)
```

#### 公式 19: 規範化波動性值
```
volatility_value = min(volatility_score, 1.0)

其中:
- volatility_value ∈ [0, 1]

解釋:
- > 0.6: 高波動性 (適合交易)
- > 0.4: 中低波動性
- < 0.3: 低波動性 (盤整狀態)
```

**代碼位置:** `indicators_implementation.py` 第 180 行
```python
volatility_value = volatility_score.clip(0, 1)
```

---

## 進階指標 (advanced_optimizer.py)

### 進階趨勢指標

#### 公式 20: 非線性趨勢值 (Sigmoid 映射)
```
trend_score = tanh(ema_ratio / 2) × (1 - exp(-atr_ratio / 0.5))
trend_value = 1 / (1 + exp(-trend_score × 3))

說明:
- 基礎公式同上
- 最後應用 Sigmoid 函數增加靈敏度
```

**代碼位置:** `advanced_optimizer.py` 第 56-58 行
```python
trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-np.maximum(atr_ratio, 0) / 0.5))
trend_value = 1 / (1 + np.exp(-trend_score * 3))
```

### 進階方向指標 (增加 MACD)

#### 公式 21: MACD 線
```
EMA12 = EMA(close, 12)
EMA26 = EMA(close, 26)
MACD = EMA12 - EMA26
```

**代碼位置:** `advanced_optimizer.py` 第 76-78 行
```python
ema12 = close_series.ewm(span=12, adjust=False).mean().values
ema26 = close_series.ewm(span=26, adjust=False).mean().values
macd = ema12 - ema26
```

#### 公式 22: MACD 信號線
```
signal_line = EMA(MACD, 9)
```

**代碼位置:** `advanced_optimizer.py` 第 79 行
```python
signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean().values
```

#### 公式 23: MACD 柱狀圖
```
MACD_histogram = MACD - signal_line
```

**代碼位置:** `advanced_optimizer.py` 第 80 行
```python
macd_histogram = macd - signal_line
```

#### 公式 24: 加權方向分數 (含 MACD)
```
direction_score = rsi_signal × 0.3 + roc_signal × 0.3 + macd_signal × 0.4

說明:
- RSI: 30% 權重
- ROC: 30% 權重
- MACD: 40% 權重 (最重要)
```

**代碼位置:** `advanced_optimizer.py` 第 91 行
```python
direction_score = (rsi_signal * 0.3 + roc_signal * 0.3 + macd_signal * 0.4)
```

### 進階波動性指標

#### 公式 25: 組合波動性分數
```
combined_vol = volatility_score × 0.6 + (high_low_range × 10) × 0.4

說明:
- volatility_score: 60% 權重
- high_low_range: 40% 權重
```

**代碼位置:** `advanced_optimizer.py` 第 119 行
```python
combined_vol = volatility_score * 0.6 + (high_low_range * 10) * 0.4
```

---

## 集合信號

### 公式 26: 集合信號組合
```
combined = trend × trend_weight + direction × direction_weight + volatility × volatility_weight × 0.5

說明:
- 波動性作為調節因子 (×0.5)
- 權重由優化器自動學習
```

**代碼位置:** `advanced_optimizer.py` 第 163-166 行
```python
combined = (trend * trend_weight + 
           direction * direction_weight + 
           volatility * volatility_weight * 0.5)
```

### 公式 27: 動態買入閾值
```
adjusted_buy_threshold = threshold_buy + (volatility × 0.05)

說明:
- 高波動性時增加閾值
- 防止在波動時被騙進
```

**代碼位置:** `advanced_optimizer.py` 第 173 行
```python
adjusted_buy_threshold = threshold_buy + (volatility * 0.05)
```

### 公式 28: 動態賣出閾值
```
adjusted_sell_threshold = threshold_sell - (volatility × 0.05)

說明:
- 高波動性時降低賣出閾值
- 更容易止損
```

**代碼位置:** `advanced_optimizer.py` 第 174 行
```python
adjusted_sell_threshold = threshold_sell - (volatility * 0.05)
```

---

## 績效評估

### 公式 29: 準確率
```
accuracy = 正確預測數 / 總預測數

說明:
- 預測下一根 K 線漲跌的準確率
- 值域: [0, 1]
```

**代碼位置:** `advanced_optimizer.py` 第 234-244 行
```python
for i in range(len(signals) - 1):
    if signals[i] != 0:
        actual_return = self.close[i+1] - self.close[i]
        if (signals[i] == 1 and actual_return > 0) or (signals[i] == -1 and actual_return < 0):
            correct += 1
        total += 1

accuracy = correct / total
```

### 公式 30: Sharpe 比率 (風險調整回報)
```
Sharpe_ratio = (平均回報 / 標準差) × sqrt(252)

說明:
- 衡量每單位風險的回報
- 252 = 年交易天數
- > 1.0: 良好
- > 2.0: 優秀
```

**代碼位置:** `advanced_optimizer.py` 第 252-253 行
```python
mean_return = np.mean(strategy_returns)
sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
```

### 公式 31: 最大回撤 (最大虧損)
```
max_drawdown = min((cumulative - running_max) / running_max)

說明:
- 從高點到低點的最大百分比虧損
- 值域: [-1, 0]
- 越接近 0 越好
```

**代碼位置:** `advanced_optimizer.py` 第 259-261 行
```python
cumulative = np.cumprod(1 + strategy_returns)
running_max = np.maximum.accumulate(cumulative)
max_drawdown = np.min(drawdown)
```

### 公式 32: 利潤因子
```
profit_factor = 總利潤 / 總虧損

說明:
- > 1.0: 獲利
- > 2.0: 優秀
- < 1.0: 虧損
```

**代碼位置:** `advanced_optimizer.py` 第 265-266 行
```python
profit_factor = (np.sum(profits) / np.sum(losses))
```

### 公式 33: 多目標優化分數
```
combined_score = Sharpe_ratio × 0.5 + accuracy × 100 × 0.5

說明:
- Sharpe 比率: 50% (風險調整)
- 準確率: 50% (預測能力)
```

**代碼位置:** `advanced_optimizer.py` 第 296 行
```python
combined_score = (sharpe_ratio * 0.5 + accuracy * 100 * 0.5)
```

---

## 代碼位置地圖

### 文件結構
```
v2-crypto-prediction/
├── indicators_implementation.py      # 基礎三指標 + 公式 1-19
│   ├── TrendIndicator              # 趨勢指標 (公式 1-7)
│   ├── DirectionIndicator          # 方向指標 (公式 8-13)
│   └── VolatilityIndicator         # 波動性指標 (公式 14-19)
│
├── optimizer_engine.py              # 基礎優化引擎
│   └── IndicatorOptimizer          # 參數優化
│
├── advanced_optimizer.py            # 進階優化引擎
│   └── AdvancedIndicatorOptimizer   # 公式 20-33
│       ├── calculate_trend_advanced        # 公式 20
│       ├── calculate_direction_advanced    # 公式 21-24
│       ├── calculate_volatility_advanced   # 公式 25
│       ├── generate_ensemble_signals      # 公式 26-28
│       └── calculate_metrics              # 公式 29-32
│
└── FORMULAS_REFERENCE.md            # 本文檔
```

### 快速查找表

| 公式編號 | 名稱 | 文件 | 行號 |
|---------|------|------|------|
| 1 | EMA | indicators_implementation.py | 52-53 |
| 2 | True Range | indicators_implementation.py | 55-59 |
| 3 | ATR | indicators_implementation.py | 61 |
| 4 | EMA 比率 | indicators_implementation.py | 64 |
| 5 | ATR 比率 | indicators_implementation.py | 68 |
| 6 | 趨勢分數 | indicators_implementation.py | 71 |
| 7 | 規範化趨勢值 | indicators_implementation.py | 74 |
| 8 | RSI | indicators_implementation.py | 114-115 |
| 9 | ROC | indicators_implementation.py | 118 |
| 10 | RSI 信號 | indicators_implementation.py | 121 |
| 11 | ROC 信號 | indicators_implementation.py | 125 |
| 12 | 方向分數 | indicators_implementation.py | 128 |
| 13 | 規範化方向值 | indicators_implementation.py | 131 |
| 14 | SMA | indicators_implementation.py | 163 |
| 15 | STDEV | indicators_implementation.py | 166 |
| 16 | Bollinger Bands | indicators_implementation.py | 169-170 |
| 17 | 波動性比率 | indicators_implementation.py | 174 |
| 18 | 波動性分數 | indicators_implementation.py | 177 |
| 19 | 規範化波動性值 | indicators_implementation.py | 180 |
| 20 | 非線性趨勢值 (Sigmoid) | advanced_optimizer.py | 56-58 |
| 21 | MACD 線 | advanced_optimizer.py | 76-78 |
| 22 | MACD 信號線 | advanced_optimizer.py | 79 |
| 23 | MACD 柱狀圖 | advanced_optimizer.py | 80 |
| 24 | 加權方向分數 | advanced_optimizer.py | 91 |
| 25 | 組合波動性分數 | advanced_optimizer.py | 119 |
| 26 | 集合信號組合 | advanced_optimizer.py | 163-166 |
| 27 | 動態買入閾值 | advanced_optimizer.py | 173 |
| 28 | 動態賣出閾值 | advanced_optimizer.py | 174 |
| 29 | 準確率 | advanced_optimizer.py | 234-244 |
| 30 | Sharpe 比率 | advanced_optimizer.py | 252-253 |
| 31 | 最大回撤 | advanced_optimizer.py | 259-261 |
| 32 | 利潤因子 | advanced_optimizer.py | 265-266 |
| 33 | 多目標優化分數 | advanced_optimizer.py | 296 |

---

## 使用指南

### 查看特定公式

1. **如果你想查看公式 6 (趨勢分數)**
   - 打開: `indicators_implementation.py`
   - 轉到: 第 71 行
   - 查看: `trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-atr_ratio / 0.5))`

2. **如果你想查看公式 24 (加權方向分數)**
   - 打開: `advanced_optimizer.py`
   - 轉到: 第 91 行
   - 查看: `direction_score = (rsi_signal * 0.3 + roc_signal * 0.3 + macd_signal * 0.4)`

3. **如果你想查看公式 33 (多目標優化分數)**
   - 打開: `advanced_optimizer.py`
   - 轉到: 第 296 行
   - 查看: `combined_score = (sharpe_ratio * 0.5 + accuracy * 100 * 0.5)`

---

## 公式修改建議

### 如果要改進趨勢指標準確率:
1. 修改公式 4 的 EMA 周期 (目前: 23/90)
2. 修改公式 5 的 ATR 周期 (目前: 10)
3. 修改公式 6 中的 tanh 和 exp 的係數

### 如果要改進方向指標準確率:
1. 修改公式 8 的 RSI 周期 (目前: 10)
2. 修改公式 9 的 ROC 周期 (目前: 8)
3. 修改公式 24 的權重 (目前: RSI 30%, ROC 30%, MACD 40%)

### 如果要改進波動性指標準確率:
1. 修改公式 14 的 SMA 周期 (目前: 39)
2. 修改公式 16 的標準差倍數 (目前: 2.6)
3. 修改公式 25 的權重 (目前: 60% 波動性, 40% 高低範圍)

---

最後更新: 2025-12-31
