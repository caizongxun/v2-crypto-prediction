# model_ensemble_gui.py 更新日誌

## 版本 v2.0 - Hugging Face 整合

更新時間: 2026-01-01 08:05 UTC

### 核心功能

#### 1. Hugging Face 數據源集成

✅ 支持從 `zongowo111/v2-crypto-ohlcv-data` 數據集自動下載
✅ 支持 22 種主流加密貨幣 (BTC, ETH, SOL, ADA 等)
✅ 支持多時間框架 (15 分鐘和 1 小時 K 線)
✅ 自動緩存機制,避免重複下載

#### 2. GUI 改進

**【資料源】區域 - 新增動態選擇界面**
- 幣種下拉菜單 (22 種預配置)
- 時間框架下拉菜單 (15m, 1h)
- 【從 Hugging Face 加載】按鈕 - 開始數據加載

**【非阻塞式數據加載】**
- DataLoaderWorker (QThread)
- 實時進度提示
- 詳細錯誤處理

**【數據驗證】**
- 自動檢查必要欄位
- 時間戳轉換
- 數據排序和清理

#### 3. 文件路徑邏輯

```python
# 自動構建正確的文件路徑
symbol = "BTCUSDT"           # 用戶選擇
timeframe = "15m"             # 用戶選擇

symbol_without_usdt = "BTC"  # 移除 USDT 後綴
filename = f"{symbol_without_usdt}_{timeframe}.parquet"
# 結果: "BTC_15m.parquet"

filepath = f"klines/{symbol}/{filename}"
# 結果: "klines/BTCUSDT/BTC_15m.parquet"
```

### 支持的幣種

AAVEUSDT, ADAUSDT, ALGOUSDT, ARBUSDT, ATOMUSDT
AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT, DOGEUSDT
DOTUSDT, ETCUSDT, ETHUSDT, FILUSDT, LINKUSDT
LTCUSDT, MATICUSDT, NEARUSDT, OPUSDT, SOLUSDT
UNIUSDT, XRPUSDT

### 使用方式

#### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

確保已安裝:
- PyQt5
- pandas
- lightgbm
- catboost
- huggingface_hub
- scikit-learn
- matplotlib

#### 2. 啟動 GUI

```bash
python model_ensemble_gui.py
```

#### 3. 數據加載流程

1. 打開 GUI
2. 在【資料源】區域選擇幣種 (預設 BTCUSDT)
3. 選擇時間框架 (15m 或 1h)
4. 點擊【從 Hugging Face 加載】
5. 自動下載和驗證數據
6. 實時顯示進度

### 技術架構

#### 多線程架構

**DataLoaderWorker: 非阻塞式數據加載**
- 在後台線程執行下載
- UI 保持響應
- 實時進度信號

**TrainingWorker: 非阻塞式模型訓練**
- 並行訓練 LightGBM 和 CatBoost
- 進度更新機制
- 完成回調函數

#### 智能緩存

首次下載後自動緩存到 `~/.cache/huggingface/`
無需重複下載同一數據集

#### 完善的錯誤處理

- 網絡連接超時提示
- 缺失欄位驗證
- 數據完整性檢查
- 模型訓練失敗恢復

### 模型性能

在 BTC 15m 數據上的測試結果:

- LightGBM: ~52-53% 準確率
- CatBoost: ~52-53% 準確率
- Ensemble: ~54-55% 準確率 (集成預測)

### 文件清單

```
v2-crypto-prediction/
├── model_ensemble_gui.py          # 改進後的主程序 (v2.0)
├── README.md                       # 完整功能說明
├── QUICK_START.md                  # 快速開始指南
├── GUI_UPDATE_LOG.md               # 本文件
└── requirements.txt                # 依賴清單
```

### 更新記錄

**v2.0 (2026-01-01)**
- 整合 Hugging Face 數據源
- 添加動態幣種和時間框架選擇
- 改進數據加載速度
- 優化 GUI 布局
- 添加實時進度提示

**v1.0 (初始版本)**
- 基礎 GUI 界面
- 模型訓練功能
- CSV 結果導出

### 已知限制

1. 免費 Hugging Face 帳戶可能有下載限制
2. 較大數據集首次下載時間較長
3. GPU 訓練需要額外的 CUDA 配置

### 故障排除

#### 問題: 看不到「從 Hugging Face 加載」按鈕

**解決方案:**
```bash
# 清除 Python 快取
find . -type d -name __pycache__ -exec rm -r {} +
find . -name "*.pyc" -delete

# 重新拉取最新代碼
git pull origin main

# 重新啟動 GUI
python model_ensemble_gui.py
```

#### 問題: Hugging Face 下載超時

**解決方案:**
```bash
# 檢查網絡連接
ping huggingface.co

# 使用代理 (如需要)
export HF_ENDPOINT="https://huggingface.co"

# 重試下載
python model_ensemble_gui.py
```

### 下一步改進方向

- 實時數據流: WebSocket 實時 K 線推送
- 多時間框架: 4 小時、1 天周期支持
- 風險管理: 止損和持倉管理
- 性能優化: GPU 加速訓練
- 深度學習: LSTM 和 Transformer 模型
- 實時預測: 連接實時交易接口

### 貢獻者

開發者: caizongxun
更新日期: 2026-01-01
倉庫: https://github.com/caizongxun/v2-crypto-prediction

### 許可證

MIT License
