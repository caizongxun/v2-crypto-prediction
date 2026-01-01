# 故障排除指南

## 問題 1: PyQt5 DLL 加載失敗

### 錯誤信息
```
ImportError: DLL load failed while importing QtWidgets: 找不到指定的程序。
```

### 根本原因

Windows 上 PyQt5 依賴於多個 C++ 運行時 DLL，如果以下任何一個缺失或版本不匹配，就會出現此錯誤：
- Visual C++ Redistributable
- Qt5 核心庫
- OpenGL 庫

### 解決方案 (按優先級排列)

#### 方案 1: 自動降級 (推薦)

**最好的方式 - 程序會自動檢測並使用 Tkinter**

```bash
python model_ensemble_gui.py
```

系統會自動嘗試加載 PyQt5，如果失敗會自動切換到 Tkinter 版本。

#### 方案 2: 修復 PyQt5 (可選)

如果您確實想使用 PyQt5，請嘗試以下步驟：

**步驟 1: 安裝 Visual C++ Redistributable**

下載並安裝：
- [Visual C++ 2015-2022 Redistributable](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)

**步驟 2: 重新安裝 PyQt5**

```bash
# 完全卸載
pip uninstall PyQt5 PyQtWebEngine -y

# 清除 pip 緩存
pip cache purge

# 重新安裝（不使用緩存）
pip install --no-cache-dir PyQt5==5.15.7
```

**步驟 3: 驗證安裝**

```bash
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 正常')"
```

#### 方案 3: 使用 Anaconda (備選)

如果上述方法都不生效，使用 Anaconda 通常能解決 DLL 依賴問題：

```bash
# 安裝 Anaconda 後
conda create -n crypto python=3.9
conda activate crypto
conda install -c conda-forge pyqt5
pip install pandas numpy huggingface-hub pyarrow
python model_ensemble_gui.py
```

### 系統設計

程序已經內置自動降級機制：

```python
use_pyqt5 = True
try:
    from PyQt5.QtWidgets import ...
    logger.info("成功加載 PyQt5")
except (ImportError, OSError) as e:
    logger.warning(f"無法加載 PyQt5: {str(e)}")
    logger.info("將使用 Tkinter 替代方案")
    use_pyqt5 = False
```

### PyQt5 vs Tkinter

| 功能 | PyQt5 | Tkinter |
|------|-------|----------|
| 外觀 | 現代化，類似 Qt | 簡樸，原生風格 |
| 性能 | 優秀 | 足夠 |
| 依賴 | C++ 運行時 | 內置 Python |
| Windows DLL | 經常出問題 | 無此問題 |
| 數據加載 | 后台線程 | 后台線程 |
| 功能完整性 | 100% | 100% |

**結論**: 兩個版本功能完全相同，只是界面風格略有不同。

---

## 問題 2: HuggingFace 下載失敗

### 錯誤信息
```
FileNotFoundError: klines/BTCUSDT/BTC_15m.parquet
ConnectionError: 無法連接到 HuggingFace
```

### 解決方案

#### 檢查網絡連接
```bash
python -c "import requests; print(requests.get('https://huggingface.co').status_code)"
```

#### 檢查文件名
確保幣種符號正確（區分大小寫）：
```python
SUPPORTED_SYMBOLS = [
    "AAVEUSDT", "ADAUSDT", "ALGOUSDT",  # ... 等
]
```

#### 設置 HuggingFace 緩存

首次下載會較慢，之後會使用本地緩存：
```bash
# 緩存位置
# Windows: C:\Users\{username}\.cache\huggingface\hub
# Linux: ~/.cache/huggingface/hub
```

#### 手動設置 HF 緩存目錄
```python
import os
os.environ['HF_HOME'] = 'D:\\my_hf_cache'  # 自定義目錄
```

---

## 問題 3: pandas 數據類型轉換錯誤

### 錯誤信息
```
ValueError: unable to parse string "" as a floating point value
```

### 原因
數據中包含空值或非數值字符串

### 解決方案

代碼已內置處理機制：
```python
df[col] = pd.to_numeric(df[col], errors='coerce')
# 錯誤值自動轉為 NaN
```

UI 會顯示 "N/A" 而不是崩潰。

---

## 問題 4: 表格加載緩慢

### 原因
- 首次下載大型 parquet 文件
- 網絡延遲
- 本地磁盤性能

### 解決方案

1. **耐心等待首次加載** (通常 10-30 秒)
2. **後續加載會使用本地緩存** (通常 < 1 秒)
3. **清空緩存重新加載**:
   ```python
   fetcher.clear_cache(symbol="BTCUSDT", timeframe="15m")
   ```

---

## 問題 5: 時間顯示不正確

### 原因
時間戳格式不同或時區問題

### 解決方案

代碼自動處理多種格式：
```python
if df['time'].dtype == 'object':
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
elif df['time'].dtype in ['int64', 'float64']:
    df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
```

---

## 問題 6: 內存不足

### 症狀
程序變慢或崩潰，顯示 "MemoryError"

### 解決方案

1. **清空舊的緩存數據**
   ```python
   fetcher.clear_cache()
   ```

2. **只查詢必要的幣種** (避免一次加載所有幣種)

3. **升級到更高配置的機器**

---

## 問題 7: 在企業網絡中無法連接

### 原因
代理設置或防火牆

### 解決方案

設置代理：
```bash
pip install --proxy [user:passwd@]proxy.server:port PyQt5
```

或者在代碼中設置：
```python
import os
os.environ['HTTP_PROXY'] = 'http://proxy:port'
os.environ['HTTPS_PROXY'] = 'https://proxy:port'
```

---

## 調試技巧

### 啟用詳細日誌

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,  # 改為 DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),  # 保存到文件
        logging.StreamHandler()  # 輸出到控制台
    ]
)
```

### 檢查 Python 環境

```bash
python --version
pip list | grep -i pyqt
pip list | grep -i pandas
```

### 測試 HuggingFace 連接

```python
from huggingface_hub import hf_hub_download
try:
    file = hf_hub_download(
        repo_id="zongowo111/v2-crypto-ohlcv-data",
        filename="klines/BTCUSDT/BTC_15m.parquet",
        repo_type="dataset"
    )
    print(f"成功: {file}")
except Exception as e:
    print(f"失敗: {e}")
```

---

## 常見問題 (FAQ)

### Q: 我應該使用 PyQt5 還是 Tkinter?
A: 都可以，功能相同。PyQt5 界面更漂亮，Tkinter 更穩定。系統會自動選擇。

### Q: 緩存文件在哪裡?
A: Windows 通常在 `C:\Users\{username}\.cache\huggingface\hub`

### Q: 可以離線使用嗎?
A: 不行，需要網絡連接下載數據。但下載後可以使用 `clear_cache=False` 使用本地緩存。

### Q: 如何更新數據?
A: 調用 `clear_cache()` 後重新加載，會重新下載最新數據。

### Q: 支持實時數據嗎?
A: 目前只支持歷史 K 線數據。實時數據需要 WebSocket 連接。

### Q: 可以添加更多幣種嗎?
A: 可以，在 `SUPPORTED_SYMBOLS` 列表中添加，前提是 HuggingFace 數據集中有該幣種。

---

## 需要更多幫助?

1. 檢查日誌文件中的詳細錯誤信息
2. 在 GitHub Issues 中提交問題
3. 提供完整的錯誤堆棧跟蹤
4. 說明 Python 版本和操作系統
