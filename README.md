# V2 Crypto Prediction System

虛擬貨幣預測系統 - BTC 15分鐘時間框架預測

## 項目描述

此項目使用程序化的一算法，約等于股票技术分析，希望把数个元素自动化化。

### 核心功能

- **數據加載**: 從 HuggingFace 加載 BTC 15分鐘 OHLCV 數據
- **黃金公式 V1**: 區間反轉樣式检測
  - 正延及連目樣子中檢測特定上樣、下樣樣式
  - 信忆稿能力評分
  - 买入及卖出信號產生

## 項目結構

```
v2-crypto-prediction/
├── config/
│   ├── constants.py          # 配置定整数 及設定
│   ├── __init__.py
├── data/
│   ├── data_loader.py        # 數據加載模組
│   ├── __init__.py
├── formulas/
│   ├── golden_formula_v1.py  # 黃金公式 V1
│   ├── __init__.py
├── requirements.txt       # Python 依賴檔案
├── README.md            # 此檔案
└── .gitignore
```

## 安裝

### 1. 克隆項目

```bash
git clone https://github.com/caizongxun/v2-crypto-prediction.git
cd v2-crypto-prediction
```

### 2. 安裝依賴檔案

```bash
pip install -r requirements.txt
```

### 3. 設定 HuggingFace Token

```bash
export HF_TOKEN='your_token_here'
```

## 使用方法

### 例子 1: 加載數據

```python
from data import load_btc_data
import os

hf_token = os.getenv('HF_TOKEN')
df = load_btc_data(
    hf_token=hf_token,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

print(f'Loaded {len(df)} records')
print(df.head())
```

### 例子 2: 使用黃金公式

```python
from data import load_btc_data
from formulas import GoldenFormulaV1
import os

hf_token = os.getenv('HF_TOKEN')
df = load_btc_data(hf_token=hf_token)

formula = GoldenFormulaV1(lookback_period=20)
patterns = formula.detect_interval_reversal(df, min_pattern_strength=0.7)

print(f'Found {len(patterns)} patterns')
for pattern in patterns[-5:]:
    print(f'{pattern.timestamp}: {pattern.signal.value} (confidence: {pattern.confidence:.2f})')

summary = formula.get_patterns_summary()
print(f'\nSummary:')
print(f'  Total: {summary["total"]}')
print(f'  Buy: {summary["buy_signals"]}')
print(f'  Sell: {summary["sell_signals"]}')
print(f'  Avg Confidence: {summary["average_confidence"]:.2f}')
```

## 黃金公式 V1 原理

### 樣式偵測

黃金公式 V1 中检測 K 線 颜色 反轉。

- **黑色 K 線**: close < open (下跌)
- **白色 K 線**: close > open (上漲)

即: 在一个約 20 根 K 線的视野汇中，列 K 線 是黑色，下一根 K 線 是白色】3 根 K 线以上本程序不投上一汇。

### 信忆稿法首天

信忆稿根據:

1. **K 線颜色一致性** (30%)
2. **体量强动** (30%)
3. **价格范围** (40%)

## 管晲性

配置文件位於 `config/constants.py`

## License

MIT
