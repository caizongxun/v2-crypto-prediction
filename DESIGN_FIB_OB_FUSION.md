# Fibonacci Bollinger Bands + Order Block 融合系統設計文檔

## 1. 整體架構

本系統通過融合 Fibonacci Bollinger Bands (FBB) 和 Order Blocks (OB) 兩個技術指標，
學習它們之間的關係，以期提升交易信號的準確性。

### 系統三層架構

```
┌─────────────────────────────────────────┐
│  Model Ensemble GUI (model_ensemble_gui.py)
│  - 可視化展示
│  - 參數調整
│  - 分析結果
└──────────────┬──────────────────────────┘
               │
       ┌───────▼────────┐
       │ FibOBFusion    │
       │ (融合分析器)    │
       └───────┬────────┘
               │
     ┌─────────┴─────────┐
     │                   │
  ┌──▼──┐          ┌─────▼──┐
  │ FBB │          │OB Detector
  │     │          │
  └─────┘          └─────────┘
```

## 2. 核心指標原理

### 2.1 Fibonacci Bollinger Bands (FBB)

**基本計算邏輯：**

```python
basis = VWMA(hlc3, length)          # 中心線：體積加權移動平均
dev = StdDev(hlc3, length) * mult   # 標準差倍數

# Fibonacci比率帶
upper/lower_0.236 = basis ± (0.236 * dev)
upper/lower_0.382 = basis ± (0.382 * dev)
upper/lower_0.5   = basis ± (0.5 * dev)
upper/lower_0.618 = basis ± (0.618 * dev)  # 最常用
upper/lower_0.764 = basis ± (0.764 * dev)
upper/lower_1.0   = basis ± (1.0 * dev)    # 上下限
```

**關鍵變數：**

| 變數 | 含義 | 預設值 | 範圍 |
|------|------|--------|-------|
| length | VWMA週期 | 200 | 50-500 |
| mult | 標準差倍數 | 3.0 | 0.5-10.0 |
| 0.618 | Fibonacci比率 | - | 最關鍵水平 |
| 1.0 | 上下限 | - | 反轉區域 |

**視覺含義：**

- **紫色線（Basis）**：中心軸，代表均衡點
- **白色線（0.236-0.764）**：支持/阻力區間
- **紅色線（Upper 1.0）**：看跌反轉上限
- **綠色線（Lower 1.0）**：看漲反轉下限

### 2.2 Order Block Detection (OB)

**檢測邏輯：**

```python
# 看漲OB（Bullish OB）：
  1. 最後一根下跌蠟燭（close < open）
  2. 接著 N 根連續上漲蠟燭（close > open）
  3. 漲幅達到門檻值 (%)
  => 該下跌蠟燭的 Open-Low 範圍為 OB

# 看跌OB（Bearish OB）：
  1. 最後一根上漲蠟燭（close > open）
  2. 接著 N 根連續下跌蠟燭（close < open）
  3. 跌幅達到門檻值 (%)
  => 該上漲蠟燭的 High-Open 範圍為 OB
```

**關鍵變數：**

| 變數 | 含義 | 預設值 | 說明 |
|------|------|--------|-------|
| periods | 必要連續根數 | 5 | 機構行為的最小週期 |
| threshold | 最小漲跌幅% | 0.0 | 驗證 OB 有效性的門檻 |
| use_wicks | 使用實體還是燈芯 | False | False=Open/Low/High；True=Full Range |

**OB 的三個關鍵水平：**

1. **High**：OB上限（看跌時的高點）
2. **Avg**：OB中點（機構入場的均衡點）
3. **Low**：OB下限（看漲時的低點）

## 3. 融合關係學習

### 3.1 核心假設

**當以下條件同時滿足時，交易信號強度增強：**

1. K線與 Fibonacci 水平相互作用（接觸、刺穿、回落）
2. 附近 50 根 K線內存在有效的 Order Block
3. OB 方向與價格相對於 Basis 的位置一致
4. OB 的中點（Avg）與當前價格距離接近

### 3.2 信號強度計算

```python
signal_strength = 0

# 因子 1：Fibonacci 水平交互數（0-30分）
interaction_count = len(fib_levels_touched)
signal_strength += min(interaction_count * 20, 30)

# 因子 2：附近 OB 數量（0-30分）
nearby_ob_count = len(obs_within_50bars)
signal_strength += min(nearby_ob_count * 15, 30)

# 因子 3：OB 方向對齐（0-20分）
if (price < basis and bearish_obs_nearby) or \
   (price > basis and bullish_obs_nearby):
    signal_strength += 20

# 因子 4：OB 中點接近度（0-20分）
nearest_ob = find_closest_ob()
if abs(nearest_ob.avg - close) / close < 0.01:  # < 1%
    signal_strength += 20
elif abs(nearest_ob.avg - close) / close < 0.03:  # < 3%
    signal_strength += 10

final_signal = min(signal_strength, 100)
```

## 4. 視覺化設計

### 4.1 融合圖表組成

```
Fibonacci-OB 融合分析圖
│
├─ K線蠟燭圖（綠紅色）
│  ├─ 綠色蠟燭：上漲
│  └─ 紅色蠟燭：下跌
│
├─ Fibonacci Bollinger Bands
│  ├─ 紫色線：VWMA Basis
│  ├─ 白色線：0.618 bands（支持/阻力）
│  ├─ 紅色線：Upper 1.0（看跌限位）
│  └─ 綠色線：Lower 1.0（看漲限位）
│
├─ Order Blocks
│  ├─ 藍色框 & v 符號：看跌 OB
│  └─ 綠色框 & ^ 符號：看漲 OB
│
└─ 融合信號標記
   ├─ 信號強度變色
   └─ 觸發點提示
```

### 4.2 圖表參數

```python
# 顯示範圍
display_bars = min(300, total_bars)  # 最後 300 根 K線

# K線繪製
color = '#00AA00' if close >= open else '#CC0000'
line_width = 0.8
bar_width = 0.6

# FBB 繪製
basis: linewidth=2, color='#FF00FF' (紫)
upper/lower_618: linewidth=1, color='#CCCCCC', alpha=0.7
upper_1: linewidth=2, color='#FF0000', alpha=0.7
lower_1: linewidth=2, color='#00AA00', alpha=0.7

# OB 繪製
rect: width=1, height=OB_range, alpha=0.25
bearish_color = '#4169E1' (深藍)
bullish_color = '#32CD32' (草綠)
```

## 5. 特徵工程思路

### 5.1 當前實現的特徵

1. **FBB 交互特徵**
   - 與各 Fibonacci 水平的交互類型（touch/near）
   - 與 Basis 的距離
   - 價格相對於 0.618 帶的位置

2. **OB 特徵**
   - OB 方向（bullish/bearish）
   - OB 距離（最近 OB 的距離）
   - OB 強度（percent_move）

3. **融合特徵**
   - Fibonacci-OB 接近度
   - 同向強度
   - 多指標確認

### 5.2 未來擴展方向

```python
# 可以加入的更多指標關聯

1. RSI + FBB：
   - RSI 極值與 Fibonacci 限位配合
   - 過買過賣確認

2. MACD + OB：
   - 動量轉換與 OB 形成的時間關係
   - 機構進出識別

3. Volume Profile + FBB：
   - 成交量集中區與 Fibonacci 帶疊合
   - 支持/阻力確認

4. Market Structure + OB：
   - 高低點遞推與 OB 位置關係
   - 趨勢方向確認

5. Machine Learning Layer：
   - XGBoost/LightGBM 融合多特徵
   - 學習 Fibonacci-OB-Price 的非線性關係
```

## 6. 使用流程

### 6.1 GUI 操作步驟

```
1. 「數據加載」tab
   ├─ 加載本地 CSV/Parquet
   └─ 或加載預設數據 (data/btc_15m.parquet)

2. 「Fibonacci OB融合」tab
   ├─ 調整參數：
   │  ├─ FBB 週期 (50-500)
   │  ├─ FBB 倍數 (0.5-10.0)
   │  └─ OB 連續根數 (3-20)
   ├─ 點擊「分析融合」
   └─ 查看可視化圖表

3. 解讀結果
   ├─ K線走勢
   ├─ FBB 水平作用
   ├─ OB 位置
   └─ 整體信號強度
```

### 6.2 參數調整指南

**FBB 週期選擇：**
- 15m K線：200-300（日線級別）
- 1h K線：100-200（4小時級別）
- 4h K線：50-100（日線級別）

**FBB 倍數選擇：**
- 保守：1.5-2.0（帶寬窄，靈敏度高）
- 標準：2.5-3.5（平衡，推薦）
- 激進：4.0-5.0（帶寬寬，波動性強）

**OB 週期選擇：**
- 更敏感：3-4（發現更多 OB，可能虛假）
- 標準：5-6（平衡，推薦）
- 更保守：7-10（只捕捉主要 OB）

## 7. 文件結構

```
project/
├── fib_ob_fusion.py
│   ├── FibonacciBollingerBands      # FBB 計算
│   ├── OrderBlockDetector            # OB 檢測
│   └── FibOBFusion                   # 融合分析
│
├── model_ensemble_gui.py
│   ├── SmartMoneyStructure           # SMC 分析（備用）
│   └── ModelEnsembleGUI              # 可視化 GUI
│
└── DESIGN_FIB_OB_FUSION.md          # 本文檔
```

## 8. 下一步計劃

### 8.1 模型訓練階段

我們將使用提取的 Fibonacci-OB 融合特徵，
訓練以下模型來預測價格方向：

1. **XGBoost**：非線性關係學習
2. **LightGBM**：快速訓練和推理
3. **LSTM**：序列依賴關係
4. **Transformer**：長期依賴捕捉

### 8.2 特徵工程擴展

逐步加入更多指標：
- RSI、MACD、Stochastic
- 成交量、波動率
- 市場結構（高低點）
- 相關資產聯動

### 8.3 回測和優化

- 參數網格搜索
- Walk-Forward 分析
- 夏普率、勝率優化
- 風險調整收益率最大化

## 9. 常見問題

**Q：為什麼選擇 Fibonacci 比率而不是其他比率？**
A：Fibonacci 比率（0.618, 1.0）是市場中最被廣泛認可的自然反轉區域。
   大量機構交易者使用，形成自我實現的預言。

**Q：OB 的 5 根連續蠟燭是否太嚴格？**
A：5 根是標準機構行為週期。可根據時間框架調整：
   - 高頻：3-4 根
   - 標準：5-6 根
   - 低頻：7-10 根

**Q：為什麼顯示最後 300 根 K線？**
A：300 根是視覺上清晰和計算效率的最佳平衡。
   - 15m K線 = 5 天左右的歷史
   - 足以識別多個 FBB 週期和 OB

**Q：信號強度 100 是什麼意思？**
A：表示 4 個主要因子全部觸發：FBB 接觸 + OB 存在 + 方向對齐 + 距離接近。
   不保證成功交易，但是最可信的聯合信號。

## 10. 參考文獻

1. 高斯分佈與標準差：用於 Bollinger Bands 的統計基礎
2. Fibonacci 序列：自然界和市場的通用法則
3. Market Microstructure：訂單簿動力學和機構行為
4. Smart Money Concepts：資金流追蹤和結構分析
