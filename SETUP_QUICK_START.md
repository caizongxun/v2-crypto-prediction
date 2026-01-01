# 快速設置指南 - 5 分鐘啟動

## 問題
```
ImportError: DLL load failed while importing QtWidgets
```

## 解決方案 (選擇一個)

### ⚡ 最快方式 (推薦)

```bash
# 1. 進入項目目錄
cd v2-crypto-prediction

# 2. 啟動虛擬環境
.venv\Scripts\activate

# 3. 自動安裝所有依賴
python install_dependencies.py

# 4. 驗證安裝
python test_dependencies.py

# 5. 運行應用
python model_ensemble_gui.py
```

**預計時間**: 3-5 分鐘

---

### 方式 2: 手動安裝

```bash
# 1. 啟動虛擬環境
.venv\Scripts\activate

# 2. 升級 pip
python -m pip install --upgrade pip

# 3. 安裝 PyQt5
pip install PyQt5==5.15.9

# 4. 安裝其他依賴
pip install pandas numpy matplotlib huggingface-hub

# 5. 運行應用
python model_ensemble_gui.py
```

**預計時間**: 2-4 分鐘

---

### 方式 3: 完整重置 (如果前兩種失敗)

```bash
# 1. 停用虛擬環境
deactivate

# 2. 刪除虛擬環境
rmdir /s /q .venv

# 3. 創建新虛擬環境
python -m venv .venv

# 4. 啟動虛擬環境
.venv\Scripts\activate

# 5. 運行自動安裝
python install_dependencies.py

# 6. 運行應用
python model_ensemble_gui.py
```

**預計時間**: 5-10 分鐘

---

## 驗證步驟

### 檢查 1: 導入測試
```bash
python -c "from PyQt5.QtWidgets import QApplication; print('✓ PyQt5 OK')"
```

### 檢查 2: 完整驗證
```bash
python test_dependencies.py
```

### 檢查 3: 啟動應用
```bash
python model_ensemble_gui.py
```

如果看到 GUI 窗口，說明設置成功！

---

## 常見問題速查

| 問題 | 解決方案 |
|------|--------|
| `DLL load failed` | 運行 `python install_dependencies.py` |
| `No module named PyQt5` | 檢查虛擬環境是否啟動 `.venv\Scripts\activate` |
| `pip not found` | 運行 `python -m pip install --upgrade pip` |
| 安裝太慢 | 使用國內鏡像: `pip install -i https://pypi.tsinghua.edu.cn/simple PyQt5==5.15.9` |
| 仍然失敗 | 參考完整指南: `INSTALL_GUIDE.md` |

---

## 應用成功啟動標志

✓ GUI 窗口出珶  
✓ 可選擇幣種  
✓ 可選擇時間框架  
✓ "加載數據" 按鐐可點擊  

---

## 下一步

1. **加載數據**
   - 選擇幣種 (例: BTCUSDT)
   - 選擇時間框架 (15m 或 1h)
   - 點擊 "加載數據"
   - 等待 30-60 秒

2. **繪製圖表**
   - 選擇 K 線數量 (50-300)
   - 勾選指標 (Fibonacci, Order Block)
   - 點擊 "繪製圖表"
   - 等待 10-15 秒

3. **查看結果**
   - 點擊 "K 線圖表" Tab
   - 查看蒺燭絡
   - 查看技術指標

---

## 獲取幫助

- 快速問題: 查看本文件
- 詳細指南: 閱讀 `INSTALL_GUIDE.md`
- 故障排除: 查看 `TROUBLESHOOTING.md`
- GitHub: https://github.com/caizongxun/v2-crypto-prediction

---

**估計時間**: 5 分鐘  
**難度**: ⚠ 簡單  
**成功率**: 95%+
