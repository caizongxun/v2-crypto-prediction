# 快速啟動指南 (Tkinter 版本)

## 為什麼改用 Tkinter?

- **無 DLL 依賴**: Tkinter 是 Python 內建模組，不需要安裝外部 DLL
- **跨平台相容**: Windows、Mac、Linux 都原生支援
- **零 Windows 問題**: 完全避免了 PyQt5 的 DLL load failed 錯誤
- **輕量級**: 應用啟動快，佔用資源少

---

## 立即啟動 (只需 1 步)

```bash
.venv\Scripts\activate
python model_ensemble_gui.py
```

**就這樣！** GUI 窗口會立即出現。

---

## 依賴情況

已安裝:
- ✅ pandas (數據處理)
- ✅ numpy (數值計算)
- ✅ huggingface-hub (數據下載)
- ✅ tkinter (內建，無需安裝)

---

## 功能說明

### 上方控制面板

1. **選擇幣種** - 下拉選單，22 種支援幣種
   - BTCUSDT (預設)
   - ETHUSDT
   - 其他主流幣種

2. **時間框架** - 選擇 K 線周期
   - 15m (15 分鐘)
   - 1h (1 小時)

3. **加載數據** - 從 HuggingFace 下載數據
   - 點擊後會顯示進度條
   - 首次下載較慢 (30-60 秒)
   - 第二次使用快取，非常快

4. **清空緩存** - 清除已下載的數據快取
   - 用於重新下載最新數據

### 中間信息面板

顯示加載的數據統計:
- 幣種名稱
- 時間框架
- 總記錄數
- 時間範圍
- 當前價格
- 24H 最高/最低

### 下方數據表格

顯示最後 100 條 K 線數據:
- 時間: K 線時間
- 開盤: 開盤價
- 最高: 最高價
- 最低: 最低價
- 收盤: 收盤價
- 成交量: 交易量

---

## 使用流程

### 第一次使用

1. 執行: `python model_ensemble_gui.py`
2. GUI 窗口彈出
3. (可選) 更改幣種和時間框架
4. 點擊 "加載數據" 按鈕
5. 等待進度條完成
6. 查看數據表格

### 後續使用

1. 執行: `python model_ensemble_gui.py`
2. 點擊 "加載數據"
3. **立即顯示** (使用快取)
4. 如需最新數據，先點擊 "清空緩存" 再加載

---

## 常見操作

### 查看比特幣 15 分鐘 K 線
```
1. 選擇幣種: BTCUSDT (預設)
2. 選擇時間框架: 15m (預設)
3. 點擊加載數據
4. 等待並查看表格
```

### 查看以太坊 1 小時 K 線
```
1. 選擇幣種: ETHUSDT
2. 選擇時間框架: 1h
3. 點擊加載數據
```

### 快速切換幣種
```
1. 從下拉選單選擇新幣種
2. 點擊加載數據
3. 使用快取，瞬間加載
```

### 更新到最新數據
```
1. 點擊清空緩存
2. 點擊加載數據
3. 等待 30-60 秒重新下載
```

---

## 故障排除

### 問題 1: "找不到 pandas"

**解決**: 
```bash
pip install pandas numpy matplotlib huggingface-hub
```

### 問題 2: 窗口無法打開

**解決**:
- 確保虛擬環境已啟動: `.venv\Scripts\activate`
- 確保 tkinter 已安裝 (Python 內建，通常無需操作)

### 問題 3: 數據加載超時

**原因**: 網路慢或 HuggingFace 服務器繁忙

**解決**:
- 等待 1-2 分鐘
- 檢查網路連接
- 嘗試清空緩存後重新加載

### 問題 4: "無法加載數據或數據為空"

**原因**: 指定幣種的數據不存在

**解決**:
- 確認幣種在支援列表中
- 嘗試其他幣種
- 檢查 HuggingFace 數據集是否可用

---

## 支援的幣種

| 幣種 | 代號 | 交易對 |
|------|------|--------|
| Aave | AAVE | AAVEUSDT |
| Cardano | ADA | ADAUSDT |
| Algorand | ALGO | ALGOUSDT |
| Arbitrum | ARB | ARBUSDT |
| Cosmos | ATOM | ATOMUSDT |
| Avalanche | AVAX | AVAXUSDT |
| Bitcoin Cash | BCH | BCHUSDT |
| Binance Coin | BNB | BNBUSDT |
| Bitcoin | BTC | BTCUSDT |
| Dogecoin | DOGE | DOGEUSDT |
| Polkadot | DOT | DOTUSDT |
| Ethereum Classic | ETC | ETCUSDT |
| Ethereum | ETH | ETHUSDT |
| Filecoin | FIL | FILUSDT |
| Chainlink | LINK | LINKUSDT |
| Litecoin | LTC | LTCUSDT |
| Polygon | MATIC | MATICUSDT |
| NEAR | NEAR | NEARUSDT |
| Optimism | OP | OPUSDT |
| Solana | SOL | SOLUSDT |
| Uniswap | UNI | UNIUSDT |
| Ripple | XRP | XRPUSDT |

---

## 技術詳情

### 架構

```
KlineGUI (主 UI)
  ├─ KlineDataFetcher (數據獲取)
  │  └─ HuggingFace Hub (Parquet 文件)
  └─ Threading (後台加載)
```

### 數據流

```
用戶點擊加載
    ↓
啟動後台線程
    ↓
檢查快取
    ↓
若無快取，從 HuggingFace 下載
    ↓
讀取 Parquet 文件
    ↓
標準化數據格式
    ↓
保存快取
    ↓
主線程更新 UI
    ↓
顯示表格和統計
```

---

## 性能指標

| 操作 | 時間 |
|------|------|
| 啟動 GUI | <1 秒 |
| 首次加載 | 30-60 秒 |
| 使用快取加載 | <1 秒 |
| 清空緩存 | 即時 |
| 表格渲染 (100 行) | <1 秒 |

---

## 下一步

- 整合技術分析指標 (Fibonacci, Order Block)
- 添加圖表可視化 (matplotlib)
- 實現自動更新
- 添加預警功能

---

**系統需求**:
- Python 3.8+
- Windows / Mac / Linux
- 網際網路連接
- ~500MB 磁碟空間 (快取)

**最後更新**: 2026-01-01
