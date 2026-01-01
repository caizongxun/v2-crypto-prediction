# Groq AI PineScript Converter 設置指南

## 功能說明

已將 Groq AI 驅動的 PineScript 轉 Python 轉換器整合到 `model_ensemble_gui.py` 中。

### 新增功能：
1. **PineScript Converter 標籤** - 新的 GUI 標籤頁面
2. **Groq API 整合** - 使用 Llama 3.1 70B 進行代碼轉換
3. **完整的轉換工作流程**：
   - API 金鑰管理
   - 連接測試
   - 檔案載入
   - 實時轉換
   - 結果保存

---

## 快速開始

### 第 1 步：申請 Groq API Key

1. 訪問 https://console.groq.com
2. 使用 Google/GitHub 賬戶註冊（免費）
3. 點擊 "API Keys"
4. 創建新的 API Key
5. 複製 Key 內容

### 第 2 步：安裝依賴

```bash
pip install requests
```

### 第 3 步：運行程序

```bash
python model_ensemble_gui.py
```

### 第 4 步：配置並使用轉換器

1. 打開程序後，點擊 **"PineScript Converter"** 標籤
2. 在 "Groq API Key" 欄位貼上你的 API Key
3. 點擊 **"Test Connection"** 驗證連接
4. 點擊 **"Initialize"** 初始化轉換器
5. 在左側輸入框貼上 PineScript 代碼
6. 點擊 **"Convert"** 開始轉換
7. 結果會顯示在右側輸出框
8. 點擊 **"Save Result"** 儲存轉換結果

---

## API 的關鍵優勢

✅ **完全免費** - 無限制使用，沒有配額限制
✅ **超快速** - Groq 是目前最快的推論引擎
✅ **高精度** - Llama 3.1 70B 模型專長於代碼理解
✅ **無需付款信息** - 純粹的免費 API

---

## 轉換結果格式

轉換完成後，結果會以 JSON 格式返回，包含：

```json
{
    "original_variables": {
        "variable_name": "explanation"
    },
    "python_code": "完整的 Python 代碼",
    "function_mappings": {
        "ta.sma": "pandas.Series.rolling().mean()"
    },
    "warnings": ["任何不確定的轉換"],
    "explanation": "指標邏輯說明"
}
```

### 儲存選項：
- **JSON** - 保留完整結構化信息
- **Python** - 只保存代碼部分
- **TXT** - 保存整個結果作為文本

---

## PineScript 轉換最佳實踐

### 支援的指標類型：
✅ 簡單均線指標 (MA, SMA, EMA)
✅ 動量指標 (RSI, MACD)
✅ 結構檢測 (Pivot Points, Support/Resistance)
✅ Order Block 檢測
✅ Custom 邏輯指標

### 轉換注意事項：
⚠️ 複雜的 PineScript 可能需要手動調整
⚠️ 某些內建函數可能無法 1:1 轉換
⚠️ 大型指標（>1000 行）可能需要分段轉換

---

## 常見問題

### Q: API Key 安全嗎？
A: 完全安全。API Key 只用於本地 API 調用，不會被存儲或傳輸到其他地方。

### Q: 轉換失敗怎麼辦？
A: 
1. 確保 API Key 有效
2. 確保 PineScript 語法正確
3. 檢查網絡連接
4. 查看錯誤信息進行調試

### Q: 可以轉換多少代碼？
A: 沒有限制。API 支持最大 4096 tokens 的輸出，可以轉換相當大的指標。

### Q: 轉換後可以直接使用嗎？
A: 大多數情況下可以。建議檢查轉換結果並進行必要的調整。

---

## 示例

### 輸入 (PineScript):
```pinescript
length = input.int(14, "Period")
ma = ta.sma(close, length)
plot(ma, "SMA", color.blue)
```

### 輸出 (Python):
```json
{
    "original_variables": {
        "length": "Moving Average period, default 14 bars",
        "ma": "Simple Moving Average of close prices"
    },
    "python_code": "import pandas as pd\ndf['sma'] = df['close'].rolling(window=14).mean()\nplot(df['sma'])",
    "function_mappings": {
        "ta.sma": "pd.Series.rolling(window=14).mean()"
    },
    "warnings": [],
    "explanation": "This is a simple moving average indicator that calculates the 14-period SMA of closing prices."
}
```

---

## 技術細節

### 使用的模型：
- **Llama 3.1 70B Versatile** - Groq 的最佳通用模型
- **推論速度** - ~400+ tokens/秒
- **上下文窗口** - 8,192 tokens

### 轉換流程：
1. 接收 PineScript 代碼
2. 構建詳細的轉換提示
3. 調用 Groq API
4. 解析返回的 JSON
5. 顯示結果

---

## 下一步

1. 測試幾個簡單指標的轉換
2. 驗證轉換結果的正確性
3. 根據需要調整和優化代碼
4. 在你的交易系統中集成轉換後的指標

---

## 支持

如有問題：
1. 檢查 API 連接狀態
2. 查看錯誤信息
3. 確認 PineScript 語法
4. 測試簡單的代碼片段

---

**祝你轉換順利！** 🚀
