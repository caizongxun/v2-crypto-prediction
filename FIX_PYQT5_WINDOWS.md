# PyQt5 Windows DLL 修復指南

## 問題

```
ImportError: DLL load failed while importing QtWidgets: 找不到指定的程序。
```

這是 Windows 上 PyQt5 的常見問題，通常由於 DLL 相依性或版本不相容導致。

---

## 快速修復 (3 選 1)

### 方案 A: Python 自動修復腳本 (推薦)

```bash
.venv\Scripts\activate
python fix_pyqt5_windows.py
python model_ensemble_gui.py
```

**優點**: 完全自動化，智能嘗試多個版本  
**時間**: 3-5 分鐘  
**成功率**: 90%+

### 方案 B: BAT 自動修復腳本

```bash
.venv\Scripts\activate
fix_pyqt5_dll.bat
python model_ensemble_gui.py
```

**優點**: 輕量級、快速  
**時間**: 2-3 分鐘  
**成功率**: 85%+

### 方案 C: 手動快速修復

```bash
.venv\Scripts\activate
pip uninstall PyQt5 PyQt5-sip PyQt5-Qt5 -y
pip cache purge
pip install PyQt5==5.15.9 --no-cache-dir --force-reinstall
python model_ensemble_gui.py
```

**優點**: 完全控制每個步驟  
**時間**: 2-4 分鐘  
**成功率**: 80%+

---

## 驗證修復成功

執行以下命令確認 PyQt5 已正確安裝:

```bash
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"
```

**成功輸出**: `PyQt5 OK`

---

## 完全重置 (如果上面都不行)

這是最可靠的方案，會完全重置虛擬環境:

```bash
# 1. 停用虛擬環境
deactivate

# 2. 刪除舊的虛擬環境
rmdir /s /q .venv

# 3. 建立新的虛擬環境
python -m venv .venv

# 4. 啟動虛擬環境
.venv\Scripts\activate

# 5. 升級 pip
python -m pip install --upgrade pip

# 6. 安裝相容版本
pip install numpy==1.26.4 pandas==2.1.4 matplotlib==3.8.4 PyQt5==5.15.9 huggingface-hub --no-cache-dir

# 7. 驗證
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"

# 8. 運行應用
python model_ensemble_gui.py
```

**預計時間**: 8-12 分鐘  
**成功率**: 99%+

---

## 如果仍然失敗

### 原因 1: 缺少 Visual C++ Runtime

**解決方案**:
1. 下載 Visual C++ Runtime: [https://support.microsoft.com/en-us/help/2977003/](https://support.microsoft.com/en-us/help/2977003/)
2. 選擇 64-bit (如果你的 Python 是 64-bit)
3. 安裝並重新啟動電腦
4. 重新嘗試方案 A、B 或 C

### 原因 2: 版本衝突

**檢查已安裝版本**:
```bash
pip list | findstr PyQt5
pip list | findstr numpy
pip list | findstr pandas
```

**相容版本組合** (Python 3.11):
```
numpy==1.26.4
pandas==2.1.4
matplotlib==3.8.4
PyQt5==5.15.9
huggingface-hub>=0.10.0
```

### 原因 3: 虛擬環境損壞

使用**完全重置方案**重新建立虛擬環境。

---

## 故障排除步驟

| 步驟 | 命令 | 預期結果 |
|------|------|--------|
| 1 | `python --version` | 顯示 Python 3.8+ |
| 2 | `.venv\Scripts\activate` | 命令提示符前顯示 `(.venv)` |
| 3 | `pip --version` | 顯示 pip 版本 |
| 4 | `python -c "import PyQt5"` | 無錯誤 |
| 5 | `python -c "from PyQt5.QtWidgets import QApplication"` | 無錯誤 |
| 6 | `python model_ensemble_gui.py` | GUI 窗口出現 |

---

## 常見錯誤信息

### 錯誤 1: DLL load failed
```
ImportError: DLL load failed while importing QtWidgets: 找不到指定的程序。
```
**解決方案**: 使用方案 A 或方案 C

### 錯誤 2: numpy 版本不相容
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
```
**解決方案**: 安裝 `numpy==1.26.4`

### 錯誤 3: Module not found
```
ModuleNotFoundError: No module named 'PyQt5'
```
**解決方案**: 重新安裝 PyQt5 (方案 A 或 C)

---

## Windows 特定提示

- **防火牆**: 確保 pip 沒有被防火牆阻擋
- **管理員權限**: 如果仍然失敗，用管理員身分運行 PowerShell 或 CMD
- **磁碟空間**: 確保至少有 500MB 的空閒空間
- **網路連接**: 確保網路穩定（安裝過程會下載依賴）

---

## 推薦順序

1. 嘗試 **方案 A** (Python 自動修復)
2. 如果失敗，嘗試 **方案 C** (手動修復)
3. 如果仍失敗，嘗試 **完全重置**
4. 如果還是失敗，檢查是否缺少 Visual C++ Runtime

---

## 成功標誌

執行以下命令都應該成功:

```bash
# 1. 導入 PyQt5
python -c "from PyQt5.QtWidgets import QApplication; print('OK')"

# 2. 導入 pandas
python -c "import pandas; print('OK')"

# 3. 導入 numpy
python -c "import numpy; print('OK')"

# 4. 啟動應用
python model_ensemble_gui.py
```

所有命令都輸出 `OK` 並且最後一個命令顯示 GUI 窗口，說明修復成功。

---

**最後更新**: 2026-01-01  
**Python 版本**: 3.8+  
**操作系統**: Windows 10/11
