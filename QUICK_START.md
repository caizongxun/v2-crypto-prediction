# 快速開始指南

## 5 分鐘快速上手

### 步驟 1: 啟動 GUI

```bash
python model_ensemble_gui_v2.py
```

### 步驟 2: 加載數據

1. 進入「Data Loading」標籤
2. 點擊「Load Default Data」(或選擇本地 CSV/Parquet)
3. 等待載入完成 ✓

### 步驟 3: 轉換指標

**三種方式:**

#### 方式 A: 使用預設指標
1. 進入「Custom Indicators」標籤
2. 點擊「Add All Default」
3. 點擊「Plot Selected」
✓ 完成!

#### 方式 B: 轉換 Pine Script
1. 進入「Indicator Converter」標籤
2. 選擇「Pine Script」
3. 貼上 Pine Script 代碼
4. 點擊「Convert Code」
5. 複製或保存輸出

#### 方式 C: 使用 Smart Money Concepts
1. 進入「Smart Money Concepts」標籤
2. 調整「Swing Length」
3. 點擊「Analyze SMC」
✓ 自動分析結構！

---

## 常見指標轉換示例

### 簡單移動平均線

```pinescript
// Pine Script
fastMA = ta.sma(close, 20)
slowMA = ta.sma(close, 50)
plot(fastMA, color=color.blue)
plot(slowMA, color=color.red)
```

↓ **自動轉換為** ↓

```python
# Python
fast_ma = close.rolling(window=20).mean()
slow_ma = close.rolling(window=50).mean()
self.add_plot('Fast MA', fast_ma, color='#0000FF')
self.add_plot('Slow MA', slow_ma, color='#FF0000')
```

### RSI

```pinescript
// Pine Script
rsi = ta.rsi(close, 14)
plot(rsi, color=color.blue)
alertcondition(rsi > 70, title="Overbought")
```

↓ **自動轉換為** ↓

```python
# Python
self.add_plot('RSI', rsi, color='#0000FF')
self.add_signal('overbought', i, rsi_value)
```

### MACD

```pinescript
// Pine Script
macd = ta.macd(close)[0]
signal = ta.macd(close)[1]
plot(macd, color=color.blue)
plot(signal, color=color.red)
```

↓ **自動轉換為** ↓

```python
# Python
self.add_plot('MACD', macd_line, color='#0000FF')
self.add_plot('Signal', signal_line, color='#FF0000')
```

---

## 分鐘級實戰流程

### 場景 1: 快速驗證指標

```
1. 打開 indicator_converter.py
2. 貼入 Pine Script
3. 點擊轉換
4. 複製輸出代碼
5. 在自己的代碼中測試
```

### 場景 2: 創建完整指標

```
1. 轉換 Pine Script → Python
2. 繼承 BaseIndicator 類
3. 實現 calculate() 方法
4. 使用 add_plot() 添加線條
5. 使用 add_signal() 添加信號
6. 在 IndicatorManager 中註冊
7. 運行和測試
```

### 場景 3: 分析市場結構

```
1. 加載數據
2. 進入 Smart Money Concepts
3. 調整 Swing Length (建議: 50)
4. 點擊 Analyze SMC
5. 查看生成的信號
```

---

## 常見問題速答

### Q: 轉換器無法識別某個函數？
**A:** 在 `indicator_converter.py` 的 `setup_mappings()` 中添加映射

### Q: 怎樣自定義顏色和線型？
**A:** 使用 `add_plot()` 的參數:
```python
self.add_plot(
    'MyLine',
    data,
    color='#FF0000',      # 十六進制顏色
    line_style='--',      # '-', '--', ':', '-.'
    width=2.0,            # 線條寬度
    alpha=0.7             # 透明度 (0-1)
)
```

### Q: 如何添加買賣信號？
**A:** 使用 `add_signal()` 方法:
```python
self.add_signal(
    'buy_signal',         # 信號類型
    i,                    # 索引
    price,                # 價格
    {'strength': 0.8}     # 可選數據
)
```

### Q: 支持其他編程語言的指標嗎？
**A:** 目前主要支持 Pine Script，可手動擴展其他語言支持

---

## 核心類和方法速查

### BaseIndicator
```python
class MyIndicator(BaseIndicator):
    def __init__(self):
        super().__init__(
            name='My Indicator',
            description='Description here',
            overlay=True/False  # True 疊在K線上, False 獨立
        )
        self.parameters = {...}
    
    def calculate(self, df: pd.DataFrame):
        # 實現邏輯
        self.add_plot('name', data, color='#000000')
        self.add_signal('signal_type', index, value)
        return {'plots': self.plots, 'signals': self.signals}
```

### IndicatorManager
```python
manager = IndicatorManager()

# 註冊
manager.register_indicator(my_indicator)

# 計算
results = manager.calculate_all(df)

# 取得信號
signals = manager.get_signals('Indicator Name')

# 繪製
fig = manager.plot_indicators(df)
```

---

## 文件結構

```
v2-crypto-prediction/
├── model_ensemble_gui_v2.py      # 主 GUI (推薦用這個)
├── indicator_converter.py         # Pine Script 轉換器
├── indicator_framework.py         # 指標框架
├── examples/
│   └── custom_indicator_example.py # 完整示例
├── INDICATOR_CONVERSION_GUIDE.md   # 詳細指南
└── QUICK_START.md                  # 本文件
```

---

## 命令行快速操作

### 運行轉換器
```bash
python indicator_converter.py
```

### 運行示例
```bash
cd examples
python custom_indicator_example.py
```

### 運行主 GUI
```bash
python model_ensemble_gui_v2.py
```

---

## 下一步

1. **深入學習**: 閱讀 `INDICATOR_CONVERSION_GUIDE.md`
2. **參考示例**: 檢查 `examples/custom_indicator_example.py`
3. **擴展功能**: 修改 `indicator_framework.py` 添加自己的邏輯
4. **性能優化**: 根據需要優化計算效率

---

## 快速排查清單

- [ ] 確認已安裝所有依賴 (pandas, numpy, matplotlib)
- [ ] 確認數據文件路徑正確
- [ ] 確認 Pine Script 代碼語法正確
- [ ] 確認 BaseIndicator 派生類實現了 calculate() 方法
- [ ] 確認 add_plot() 和 add_signal() 調用正確
- [ ] 確認顏色值是十六進制格式
- [ ] 查看控制台錯誤信息

---

**提示**: 遇到問題? 查看 INDICATOR_CONVERSION_GUIDE.md 的故障排除部分！
