# 更新說明 - HuggingFace 數據集集成

## 更新內容

### 主要改進

1. **數據來源更改**
   - 從 API 實時抓取改為從 HuggingFace 數據集讀取
   - 使用您自有的 HF 數據集: `zongowo111/v2-crypto-ohlcv-data`
   - 數據路徑格式: `klines/{symbol}/{symbol_prefix}_{timeframe}.parquet`
   - 例如: `klines/BTCUSDT/BTC_15m.parquet`

2. **PyQt5 UI 改進**
   - 后台線程加載數據（不卡 UI）
   - 實時進度顯示
   - 完整的錯誤處理
   - 數據統計信息展示

3. **本地緩存機制**
   - 自動緩存已下載的數據
   - 加快重複加載速度
   - 支持手動清除緩存

4. **支持的幣種 (23個)**
   - AAVEUSDT, ADAUSDT, ALGOUSDT, ARBUSDT, ATOMUSDT
   - AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT, DOGEUSDT
   - DOTUSDT, ETCUSDT, ETHUSDT, FILUSDT, LINKUSDT
   - LTCUSDT, MATICUSDT, NEARUSDT, OPUSDT, SOLUSDT
   - UNIUSDT, XRPUSDT

5. **支持的時間框架**
   - 15分鐘 (15m)
   - 1小時 (1h)

## 安裝步驟

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

或者逐個安裝:

```bash
pip install pandas numpy PyQt5 huggingface-hub pyarrow
```

### 2. 運行程序

```bash
python model_ensemble_gui.py
```

## 使用指南

### 基本操作流程

1. **選擇幣種**
   - 點擊「選擇幣種」下拉菜單
   - 選擇您想查看的幣種 (例如 BTCUSDT)

2. **選擇時間框架**
   - 點擊「時間框架」下拉菜單
   - 選擇 15m 或 1h

3. **加載數據**
   - 點擊「加載數據」按鈕
   - 系統會自動從 HF 下載並顯示數據
   - 進度條會顯示加載狀態

4. **查看數據**
   - 數據信息面板: 顯示幣種、時間框架、記錄數、時間範圍、當前價格等
   - K 線數據表格: 顯示最后 100 條 K 線 (時間、開盤、最高、最低、收盤、成交量)

5. **清空緩存** (可選)
   - 點擊「清空緩存」按鈕
   - 下次加載時會重新從 HF 下載

## 核心類和方法

### KlineDataFetcher 類

```python
# 初始化
fetcher = KlineDataFetcher()

# 獲取 K 線數據
df = fetcher.fetch_kline_data(symbol="BTCUSDT", timeframe="15m")

# 獲取最新 K 線
latest = fetcher.get_latest_kline(symbol="BTCUSDT")

# 獲取指定日期範圍的數據
df = fetcher.get_kline_range(
    symbol="BTCUSDT",
    timeframe="15m",
    start_date="2025-12-31",
    end_date="2026-01-01"
)

# 清空緩存
fetcher.clear_cache()
```

### DataLoadThread 類

- 后台線程用於非阻塞式加載數據
- 信號: `finished`, `error`, `data_loaded`

### ModelEnsembleGUI 類

- PyQt5 主窗口類
- 管理所有 UI 元素和用戶交互

## 數據格式說明

### 輸入列名變體 (自動轉換)

- `open_time` / `timestamp` → `time`
- `open_price` → `open`
- `high_price` → `high`
- `low_price` → `low`
- `close_price` → `close`

### 輸出列結構

| 列名 | 類型 | 說明 |
|------|------|------|
| time | datetime64 | K 線開盤時間 |
| open | float64 | 開盤價 |
| high | float64 | 最高價 |
| low | float64 | 最低價 |
| close | float64 | 收盤價 |
| volume | float64 | 成交量 |

## 故障排除

### 問題 1: 找不到 HuggingFace 數據

**解決方案:**
- 檢查網絡連接
- 確認幣種名稱正確
- 確認時間框架是 "15m" 或 "1h"

### 問題 2: PyQt5 DLL 加載失敗

**解決方案:**
```bash
pip uninstall PyQt5 -y
pip install PyQt5 --no-cache-dir
```

### 問題 3: 數據加載緩慢

**原因:** 首次下載需要時間
**解決方案:** 使用本地緩存，後續加載會更快

## 技術細節

### HuggingFace Hub 配置

```python
HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
HF_REPO_TYPE = "dataset"
```

### 文件路徑構造

```python
symbol = "BTCUSDT"
timeframe = "15m"
symbol_prefix = symbol.replace("USDT", "")  # "BTC"
filename = f"klines/{symbol}/{symbol_prefix}_{timeframe}.parquet"
# 結果: "klines/BTCUSDT/BTC_15m.parquet"
```

### 緩存機制

- 緩存鍵格式: `f"{symbol}_{timeframe}"`
- 例如: `"BTCUSDT_15m"`
- 內存存儲，程序關閉後清空

## 更新記錄

### 版本 2.0.0 (2026-01-01)

✓ 改為從 HuggingFace 數據集讀取 K 線數據
✓ 完整的 PyQt5 GUI 實現
✓ 后台線程加載機制
✓ 本地緩存支持
✓ 錯誤處理和日誌記錄
✓ 支持 23 種加密貨幣
✓ 支持 15m 和 1h 時間框架

## 後續計畫

- [ ] 添加更多技術指標計算
- [ ] 支持更多時間框架 (1m, 5m, 4h, 1d)
- [ ] 數據導出功能 (CSV, Excel)
- [ ] 實時預測功能
- [ ] 數據可視化圖表

## 聯繫和支援

如有問題或建議，歡迎提交 Issue 或 PR。
