# V2 Crypto Prediction System

一個基於 AI 逆向推理的加密貨幣交易預測系統。通過演算法生成黃金公式，結合機器學習模型預測最優開單點位。

## 系統架構

系統分為三個核心階段：

### 階段 1：黃金公式生成

AI 逆向推理基礎指標，生成三套獨立的黃金公式：

1. **趨勢強度公式 (Trend Strength)**
   - 輸出範圍：0 ~ 1
   - 含義：0 = 無趨勢，1 = 強趨勢
   - 應用：判斷當前市場是否處於明確的趨勢中

2. **波動率公式 (Volatility Index)**
   - 輸出範圍：0 ~ 1  
   - 含義：0 = 低波動，1 = 高波動
   - 應用：評估市場波動幅度，調整倉位管理

3. **方向確認公式 (Direction Confirmation)**
   - 輸出範圍：0 ~ 1
   - 含義：> 0.5 = 看多，< 0.5 = 看空，= 0.5 = 中性
   - 應用：確認市場主要方向

#### 基礎指標輸入

系統使用以下基礎指標進行組合優化：

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Moving Averages (SMA/EMA)
- Volume Profile
- Stochastic Oscillator

### 階段 2：機器學習模型訓練

基於三個黃金公式的輸出值進行特徵工程，訓練模型學習指標與開單點位的關係。

**特徵組成：**
- trend_score (0-1)
- volatility_score (0-1)
- direction_score (0-1)
- 原始基礎指標值

**目標：**
- entry_price：預測的進場點位
- entry_direction：進場方向 (多/空)
- confidence：模型信心度

### 階段 3：實時預測

將公式輸出值輸入模型，得到精確的開單建議。

**輸入範例：**
```
trend_score: 0.75 (強趨勢)
volatility_score: 0.3 (低波動)
direction_score: 0.6 (看多)
current_price: 80000
```

**預測輸出：**
```
entry_price: 79500
entry_direction: LONG
confidence: 0.87
stop_loss: 78000
take_profit: 82000
```

## 項目結構

```
v2-crypto-prediction/
├── data.py                         # 數據加載模組
├── formulas/                       # 黃金公式模組
│   ├── formula_generator.py        # 公式生成器
│   ├── trend_strength.py           # 趨勢強度公式
│   ├── volatility_index.py         # 波動率公式
│   └── direction_confirmation.py   # 方向確認公式
├── indicators/                     # 基礎指標實現
│   ├── rsi.py
│   ├── macd.py
│   ├── bollinger_bands.py
│   └── ...
├── models/                         # 機器學習模型
│   ├── trainer.py                  # 模型訓練器
│   ├── predictor.py                # 預測引擎
│   └── ensemble.py                 # 模型集成
├── backtest/                       # 回測框架
│   ├── backtester.py               # 回測引擎
│   └── metrics.py                  # 性能指標
├── config/                         # 配置文件
│   └── constants.py                # 常數定義
├── requirements.txt                # Python 依賴
└── README.md                       # 本文件
```

## 數據來源

**HuggingFace Dataset：**
- Repository ID：`zongowo111/v2-crypto-ohlcv-data`
- 文件路徑：`klines/BTCUSDT/BTC_15m.parquet`
- 本地快取：`./data/btc_15m.parquet`

**數據欄位：**
- open：開盤價
- high：最高價
- low：最低價
- close：收盤價
- volume：交易量
- timestamp：時間戳（索引）

## 環境設置

### 依賴安裝

```bash
pip install -r requirements.txt
```

### 環境變數

在項目根目錄創建 `.env` 文件：

```bash
HF_TOKEN=your_huggingface_token_here
```

或在命令行設定：

```bash
export HF_TOKEN='your_token_here'
```

## 使用方法

### 1. 加載訓練數據

```python
from data import load_btc_data
from config import HF_TOKEN

df = load_btc_data(
    hf_token=HF_TOKEN,
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

### 2. 生成黃金公式

```python
from formulas.formula_generator import FormulaGenerator

generator = FormulaGenerator(df)
formulas = generator.generate(
    num_formulas=3,
    indicators=['rsi', 'macd', 'bollinger_bands', 'atr']
)
```

### 3. 訓練預測模型

```python
from models.trainer import ModelTrainer

trainer = ModelTrainer(df, formulas)
model = trainer.train(
    model_type='ensemble',
    test_size=0.2,
    epochs=100
)
```

### 4. 進行預測

```python
from models.predictor import Predictor

predictor = Predictor(model, formulas)
prediction = predictor.predict(
    trend_score=0.75,
    volatility_score=0.3,
    direction_score=0.6,
    current_price=80000
)

print(f"進場點位：{prediction['entry_price']}")
print(f"進場方向：{prediction['entry_direction']}")
print(f"信心度：{prediction['confidence']}")
```

## 回測

```python
from backtest.backtester import Backtester

backtester = Backtester(df, predictor)
results = backtester.run(
    start_date='2024-06-01',
    end_date='2024-12-31',
    risk_reward_ratio=1.5
)

results.print_summary()
```

## 開發進度

- [ ] 階段 1：黃金公式生成與優化
  - [ ] 基礎指標實現
  - [ ] 公式生成演算法
  - [ ] 公式回測驗證
- [ ] 階段 2：機器學習模型訓練
  - [ ] 特徵工程
  - [ ] 模型選擇與訓練
  - [ ] 交叉驗證
- [ ] 階段 3：實時預測系統
  - [ ] 預測引擎
  - [ ] 回測框架
  - [ ] 實盤集成

## 交易參數

- **交易標的：** BTC/USDT
- **時間框架：** 15 分鐘
- **風險報酬比：** 1 : 1.5 (止損 1，獲利 1.5)
- **最大單筆虧損：** 待定

## 免責聲明

本系統僅用於研究和學習目的。使用本系統進行真實交易需自行評估風險。過往績效不代表未來結果。

## 聯繫方式

如有問題或建議，歡迎提交 Issue 或 Pull Request。
