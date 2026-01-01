# PyQt5 安裝指南

## 問題診斷

**錯誤**:
```
ImportError: DLL load failed while importing QtWidgets: 找不到指定的程序。
```

**原因**: PyQt5 缺少系統依賴或版本不匹配

---

## 解決方案

### 方案 1: 完整重新安裝 (推薦)

```bash
# 1. 停用虛擬環境
deactivate

# 2. 刪除虛擬環境
rmdir /s /q .venv

# 3. 重新創建虛擬環境
python -m venv .venv

# 4. 啟動虛擬環境
.venv\Scripts\activate

# 5. 升級 pip
python -m pip install --upgrade pip

# 6. 安裝所有依賴
pip install PyQt5==5.15.9
pip install pandas numpy matplotlib
pip install huggingface-hub

# 7. 驗證安裝
python -c "from PyQt5.QtWidgets import QApplication; print('✓ PyQt5 安裝成功')"
```

### 方案 2: 僅更新 PyQt5

```bash
# 啟動虛擬環境
.venv\Scripts\activate

# 卸載舊版本
pip uninstall PyQt5 -y

# 安裝新版本
pip install PyQt5==5.15.9 --force-reinstall --no-cache-dir

# 驗證
python -c "from PyQt5.QtWidgets import QApplication; print('✓ PyQt5 安裝成功')"
```

### 方案 3: 使用自動安裝腳本

```bash
# 啟動虛擬環境
.venv\Scripts\activate

# 運行自動安裝腳本
python install_dependencies.py
```

### 方案 4: 使用 conda (替代方案)

```bash
# 如果已安裝 conda
conda create -n crypto_env python=3.9
conda activate crypto_env
conda install pyqt=5.15.9
conda install pandas numpy matplotlib
pip install huggingface-hub
```

---

## 完整依賴列表

```bash
pip install --upgrade pip setuptools wheel

# 核心依賴
pip install PyQt5==5.15.9
pip install PyQt5-sip==12.13.0

# 數據處理
pip install pandas>=1.3.0
pip install numpy>=1.20.0

# 圖表渲染
pip install matplotlib>=3.4.0

# 數據源
pip install huggingface-hub>=0.10.0

# 可選: 加速
pip install scipy scikit-learn
```

---

## Windows 特定修復

如果上述方法不起作用，嘗試以下操作:

### 修復 1: Visual C++ 運行時

下載並安裝:
- [Microsoft Visual C++ Redistributable](https://support.microsoft.com/en-us/help/2977003/)

選擇對應 Python 位數的版本:
```bash
# 檢查 Python 位數
python -c "import struct; print(struct.calcsize('P') * 8)"
# 輸出 64 則選擇 x64 版本
```

### 修復 2: 清除緩存

```bash
# 清除 pip 緩存
pip cache purge

# 清除 PyQt5 相關的臨時文件
rmdir /s /q %APPDATA%\pip
rmdir /s /q %TEMP%\pip-*
```

### 修復 3: 使用 PyQt5-tools

```bash
pip uninstall PyQt5 -y
pip install PyQt5==5.15.9 --no-binary PyQt5
```

---

## 驗證安裝成功

```python
# test_install.py
import sys
print(f"Python: {sys.version}")

try:
    from PyQt5.QtWidgets import QApplication
    print("✓ PyQt5.QtWidgets 導入成功")
except ImportError as e:
    print(f"✗ PyQt5.QtWidgets 導入失敗: {e}")

try:
    import pandas as pd
    print("✓ pandas 導入成功")
except ImportError as e:
    print(f"✗ pandas 導入失敗: {e}")

try:
    import numpy as np
    print("✓ numpy 導入成功")
except ImportError as e:
    print(f"✗ numpy 導入失敗: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib 導入成功")
except ImportError as e:
    print(f"✗ matplotlib 導入失敗: {e}")

try:
    from huggingface_hub import hf_hub_download
    print("✓ huggingface_hub 導入成功")
except ImportError as e:
    print(f"✗ huggingface_hub 導入失敗: {e}")

print("\n所有依賴檢查完成!")
```

運行驗證:
```bash
python test_install.py
```

---

## 快速檢查清單

- [ ] Python 版本 >= 3.8
- [ ] 虛擬環境已啟動
- [ ] pip 已升級
- [ ] PyQt5 == 5.15.9
- [ ] pandas >= 1.3.0
- [ ] numpy >= 1.20.0
- [ ] matplotlib >= 3.4.0
- [ ] huggingface-hub >= 0.10.0
- [ ] 所有導入測試通過

---

## 常見問題

**Q: 安裝 PyQt5 時提示 "error: Microsoft Visual C++ 14.0 is required"**
A: 下載並安裝 [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**Q: 虛擬環境中 PyQt5 仍無法導入**
A: 嘗試方案 4 (使用 conda 創建環境)

**Q: PyQt5 安裝速度太慢**
A: 使用國內鏡像加速
```bash
pip install -i https://pypi.tsinghua.edu.cn/simple PyQt5==5.15.9
```

**Q: 確認安裝了但仍然報錯**
A: 運行自動安裝腳本獲取詳細診斷
```bash
python install_dependencies.py
```

---

## 完成後

一旦所有依賴安裝成功，運行:

```bash
python model_ensemble_gui.py
```

應該能看到 GUI 窗口啟動！

---

**最後更新**: 2026-01-01  
**推薦版本**: PyQt5 5.15.9 + Python 3.9/3.10
