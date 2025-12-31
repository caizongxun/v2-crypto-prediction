# 第二步：黃金公式 V2 的理論結構

## 目標

建立一個**完整的理論框架**，而不是盲目搜尋參數。核心理念是：

**Trend (40%) + Momentum (30%) + Volume (20%) + Volatility Filter (10%)**

## 理論結構圖

```
入場信號生成
    ↓
┌───────────────────────────────────────────┐
│         綜合評分系統 (Composite Score)      │
├───────────────────────────────────────────┤
│                                           │
│  趨勢 (40%)        →  多時間框架 EMA      │
│  ├─ Trend_Score                          │
│  │                                       │
│  動能 (30%)       →  RSI / Stoch / ROC   │
│  ├─ Momentum_Score                       │
│  │                                       │
│  成交量 (20%)     →  Volume Spike / VWAP  │
│  ├─ Volume_Score                         │
│  │                                       │
│  波動率 (10%)     →  ATR 篩選 (門檻)      │
│  └─ Volatility_Filter (通過/不通過)       │
│                                           │
│  最終信心分數 = 如果(波動率 OK)           │
│                Trend*0.4 + Momentum*0.3  │
│                + Volume*0.2             │
│                                           │
│  信心 >= 65% → 出場信號 (BUY/SELL)       │
└───────────────────────────────────────────┘
```

## 一、趨勢組件 (Trend Component) - 40%

### 目的
識別整體市場方向，作為操作的大框架

### 指標

#### 1.1 多時間框架 EMA
```
BTC 15分鐘 K線

EMA_15 (15根K線)  = 短期趨勢 (15 * 15分鐘 ≈ 3.75小時)
EMA_60 (60根K線)  = 長期趨勢 (60 * 15分鐘 = 15小時)

買入條件:
  EMA_15 > EMA_60  → 上升趨勢
  
賣出條件:
  EMA_15 < EMA_60  → 下降趨勢
```

#### 1.2 SuperTrend (趨勢確認)
```
SuperTrend = HL2 ± ATR * 倍數

參數:
  period = 10
  multiplier = 3.0

功能:
  - 確認 EMA 的趨勢方向
  - 提供動態止損位置
  - 當價格穿越 SuperTrend 時發出警訊
```

#### 1.3 ADX (趨勢強度過濾)
```
ADX >= 25  → 趨勢足夠強，可以操作
ADX < 25   → 趨勢太弱，可能震蕩，不操作

用途:
  - 篩選震蕩市場
  - 只在趨勢市場中交易
```

### 計算 Trend_Score

```python
if ATR < 波動率下限:
    return 0  # 波動率太低，不操作

trend_direction = 0
if EMA_15 > EMA_60 and SuperTrend == 上升:
    trend_direction = 1  # 買入信號
elif EMA_15 < EMA_60 and SuperTrend == 下降:
    trend_direction = -1  # 賣出信號
else:
    trend_direction = 0  # 無信號

trend_strength = ADX / 100  # 標準化到 0-1

Trend_Score = abs(trend_direction) * trend_strength
# 結果: 0 (無方向) 到 1 (強趨勢)
```

---

## 二、動能組件 (Momentum Component) - 30%

### 目的
確認買賣力度，避免逆勢進場

### 指標

#### 2.1 RSI (相對強弱指數)
```
RSI = 100 - (100 / (1 + RS))
RS = 上升平均 / 下降平均

區間:
  RSI < 30  → 超賣 (可能反彈)
  30 < RSI < 70  → 中性
  RSI > 70  → 超買 (可能下跌)

買入過濾: RSI > 50  (確認上升動能)
賣出過濾: RSI < 50  (確認下降動能)
```

#### 2.2 Stochastic RSI (隨機 RSI)
```
Stoch RSI = 隨機變換(RSI)

%K = 20日 RSI 最低 / (最高 - 最低)
%D = %K 的 3日 SMA

買入過濾:
  Stoch_RSI_K > Stoch_RSI_D  (動能加強)
  
賣出過濾:
  Stoch_RSI_K < Stoch_RSI_D  (動能減弱)
```

#### 2.3 ROC (變化率)
```
ROC = (Close - Close[12期前]) / Close[12期前] * 100

ROC > 0   → 價格上升 (買入力度)
ROC < 0   → 價格下降 (賣出力度)
```

### 計算 Momentum_Score

```python
# 標準化 RSI 到 0-1
rsi_signal = 1 if RSI > 50 else (-1 if RSI < 50 else 0)
rsi_strength = abs(RSI - 50) / 50  # 距中點越遠越強

# Stoch RSI 確認
stoch_confirmation = 1 if K > D else -1

# ROC 確認
roc_confirmation = 1 if ROC > 0 else (-1 if ROC < 0 else 0)

# 綜合
momentum_indicators = [rsi_signal, stoch_confirmation, roc_confirmation]
momentum_votes = sum(1 for x in momentum_indicators if x > 0)  # 投票

Momentum_Score = momentum_votes / len(momentum_indicators)
# 結果: 0 (完全空頭) 到 1 (完全多頭)
```

---

## 三、成交量組件 (Volume Component) - 20%

### 目的
確認買賣成交真實性，避免虛假信號

### 指標

#### 3.1 成交量飆升
```
Volume_SMA = Volume 的 20日 SMA

Volume_Spike = 當前成交量 > Volume_SMA * 1.5

買入過濾: 上升趨勢 + 高成交量
賣出過濾: 下降趨勢 + 高成交量
```

#### 3.2 VWAP 偏離
```
VWAP = 累計(HL2 * Volume) / 累計(Volume)

VWAP_Deviation = (Close - VWAP) / VWAP * 100

Close > VWAP * 1.01   → 價格在 VWAP 上方（買方主導）
Close < VWAP * 0.99   → 價格在 VWAP 下方（賣方主導）

買入過濾: Close > VWAP
賣出過濾: Close < VWAP
```

### 計算 Volume_Score

```python
volume_signal = 0
if current_volume > volume_sma * 1.5:
    if trend == "up":
        volume_signal = 1  # 上升成交量強
    elif trend == "down":
        volume_signal = -1  # 下降成交量強

vwap_signal = 0
if close > vwap * 1.01:
    vwap_signal = 1  # 價格在 VWAP 上方
elif close < vwap * 0.99:
    vwap_signal = -1  # 價格在 VWAP 下方

volume_votes = [volume_signal, vwap_signal]
Volume_Score = (sum(volume_votes) + 2) / 4
# 結果: 0 到 1
```

---

## 四、波動率篩選 (Volatility Filter) - 門檻

### 目的
篩選掉不適合交易的市場環境

### 指標

#### 4.1 ATR (平均真實波幅)
```
ATR = 14日 EMA(True Range)

最小波動率:
  ATR < 收盤價 * 0.5%  → 波動率太低，跳過
  
最大波動率 (可選):
  ATR > 收盤價 * 2.0%  → 可能是黑天鵝，謹慎
```

### 邏輯

```python
if ATR < close * 0.005:  # 最小波動率
    return SKIP  # 不操作，等待波動增加

if ATR > close * 0.020:  # 最大波動率 (選擇性)
    return CAUTION  # 警告，可能是異常波動

return PROCEED  # 波動率正常，繼續檢查其他條件
```

---

## 五、综合評分系統 (Composite Score)

### 最終邏輯

```python
def calculate_signal(df_row):
    # 步驟 1: 波動率篩選
    if not volatility_filter(df_row):
        return SKIP, 0
    
    # 步驟 2: 計算各組件分數
    trend_score = calculate_trend_score(df_row)
    momentum_score = calculate_momentum_score(df_row)
    volume_score = calculate_volume_score(df_row)
    
    # 步驟 3: 加權綜合
    confidence = (
        trend_score * 0.4 +
        momentum_score * 0.3 +
        volume_score * 0.2
    )
    
    # 步驟 4: 決定信號
    if confidence >= 0.65:
        if trend_score > 0.5:
            return BUY, confidence
        else:
            return SELL, confidence
    else:
        return NEUTRAL, confidence

return signal, confidence
```

---

## 六、可優化的參數

### 階段 1: 指標參數
```
EMA 周期:        fast=15, slow=60
SuperTrend:     period=10, multiplier=3.0
ADX:            period=14, min_threshold=25
RSI:            period=14
Stoch RSI:      period=14, smooth_k=3, smooth_d=3
ROC:            period=12
Volume SMA:     period=20
ATR:            period=14
```

### 階段 2: 權重優化
```
當前設定:
  Trend:       40%
  Momentum:    30%
  Volume:      20%
  Volatility:  Filter (10%)

可調節: 
  - 增加 Momentum 比重（敏銳進場）
  - 降低 Volume 比重（流動性較低的幣種）
  - 調整 Volatility 門檻（不同的風險偏好）
```

### 階段 3: 閾值優化
```
當前設定:
  Confidence 最小值:  0.65 (65%)
  RSI 中點:          50
  Stoch 超買/超賣:   80/20
  ATR 最小比例:      0.5% of close
  Volume Spike:      1.5x SMA
  VWAP 偏離:        ±1%

可調節:
  - Confidence 閾值（65% → 60% or 70%）
  - ATR 篩選（松/緊）
  - RSI/Stoch 區間（隨市場調整）
```

---

## 七、下一步行動

### 實施計畫

1. **實裝指標計算** (第二步)
   - 在 `formulas/indicators.py` 中實現所有指標
   - 創建 `IndicatorCalculator` 一鍵計算

2. **實裝配置系統** (第二步)
   - 在 `formulas/golden_formula_v2_config.py` 中定義所有參數
   - 支援輕鬆修改配置

3. **實裝公式邏輯** (第三步)
   - 實現 `GoldenFormulaV2` 類
   - 整合所有組件的評分邏輯

4. **回測框架** (第四步)
   - 計算勝率、收益率、最大回撤
   - 優化參數

5. **可視化分析** (第五步)
   - K 線圖 + 指標 + 信號標記
   - 查看假信號發生位置

---

## 八、關鍵設計原則

✅ **多時間框架確認**
  - 用 15min 和 1h 的 EMA 多重確認趨勢

✅ **指標組合（不單一指標）**
  - RSI + Stoch RSI + ROC 三角驗證
  - 避免偽信號

✅ **波動率篩選優先**
  - ATR 不達標就跳過，保護資本

✅ **加權評分制**
  - 各組件權重清晰
  - 易於調整和優化

✅ **信心分數門檻**
  - >= 65% 才出場信號
  - 避免邊界信號

---

## 文件參考

- `formulas/indicators.py` - 指標計算實現
- `formulas/golden_formula_v2_config.py` - 配置結構定義
- `STEP3_IMPLEMENTATION.md` - 實現細節 (待完成)
