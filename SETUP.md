# 本地開發設定指南

## 步驄01: 克隆項目

```bash
git clone https://github.com/caizongxun/v2-crypto-prediction.git
cd v2-crypto-prediction
```

## 步驄02: 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

## 步驄03: 設定 .env 檔案

### 方法 1: 手動創建 .env

1. 在項目根目錄創建 `.env` 檔案
2. 複制以下內容:

```bash
# HuggingFace API Token
HF_TOKEN=hf_your_token_here

# Data Configuration
START_DATE=2023-01-01
END_DATE=2025-12-31

# Trading Configuration
TRADING_PAIR=BTCUSDT
TIMEFRAME=15m

# Formula Configuration
LOOKBACK_PERIOD=20
MIN_PATTERN_STRENGTH=0.7

# Logging
LOG_LEVEL=INFO
```

3. 將 `hf_your_token_here` 替換m你的真實 HuggingFace token

### 方法 2: 使用 .env.example

```bash
cp .env.example .env
```

然後罩書 `.env` 中的 token

## 步驄04: 驗證配置

### 測試 .env 配置

```bash
python test_env.py
```

預期輸出:
```
======================================================================
當前配置
======================================================================
HF_TOKEN: 設定
交易對: BTCUSDT
時間框架: 15m
回測期間: 2023-01-01 至 2025-12-31
回看週期: 20
最小樣式強度: 0.7
日誌級別: INFO
======================================================================

配置驗證: 成功
  HF_TOKEN: 已設定
  TRADING_PAIR: BTCUSDT
  TIMEFRAME: 15m
  START_DATE: 2023-01-01
  END_DATE: 2025-12-31
  LOOKBACK_PERIOD: 20
  MIN_PATTERN_STRENGTH: 0.7
  LOG_LEVEL: INFO

所有配置成功加載！
```

### 全面测試

```bash
python test_run.py
```

預期輸出:
```
*..* V2 Crypto Prediction System - 環境測試
*..* 

======================================================================
環境検查
======================================================================
HF_TOKEN: 已設定
Python 版本: 3.10.x

======================================================================
數據加載器
======================================================================
列出可用交易對...
找到 30 個交易對:
  - BTCUSDT
  - ETHUSDT
  - BNBUSDT
  ... 還有 27 個

======================================================================
測試黃金公式 V1
======================================================================
检測樣式...
找到 5 個樣式

最近 5 個樣式:
  2024-01-01 10:45:00: BUY (信忆: 0.82)
  2024-01-01 11:00:00: SELL (信忆: 0.75)
  ...

摆要:
  總計: 5
  買入: 3
  賣出: 2
  平均信忆: 0.78

======================================================================
測試結果摆要
======================================================================
環境検查           : PASS
數據加載器        : PASS
黃金公式 V1       : PASS
======================================================================

所有測試通過！環境配置正確。
```

## PyCharm 中執行

### 方法 1: 直接執行

1. 引右擊擊 `test_run.py`
2. 選撧 **Run 'test_run'**

### 方法 2: 鍵盤快捷键

- **Windows/Linux**: `Ctrl+Shift+F10`
- **Mac**: `Ctrl+Shift+R`

## 常見問題

### 問題 1: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'config'
```

**解決方案**:

1. 確保項目根目錄在 Python Path 中
2. 在 PyCharm 中: **File → Settings → Project → Python Interpreter → Python Path**
3. 點擊 **+** 添加項目根目錄

### 問題 2: HF_TOKEN 未設定

```
警告: HF_TOKEN 環境變數未設定
```

**解決方案**:

1. 確保 `.env` 檔案存在且位於項目根目錄
2. 確保 `.env` 中有 `HF_TOKEN=` 一行
3. 重新啟動 PyCharm

### 問題 3: 依賴安裝失敗

```bash
# 手動安裝各個套件
pip install pandas numpy huggingface-hub requests scikit-learn matplotlib seaborn python-dotenv
```

## 下一步

環境配置完成後，你可以:

1. 修改 `.env` 中的配置來控制蒸気樣子検測
2. 開發你自己的測試脚本
3. 探索黃金公式策略

## 提供事項

- `.env` 檔案包含敏感信息 (你的 token)
- **不要提交 `.env` 到 GitHub**
- `.env` 已在 `.gitignore` 中，不會被貮提交
- 只有 `.env.example` 會被提交，作為配置檔案檔
