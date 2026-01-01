# 指標轉換與整合指南

本文檔說明如何將第三方指標源碼轉換為 Python，並集成到系統中使用。

## 系統組件概述

### 1. indicator_converter.py
將交易指標源碼（主要支持 Pine Script）轉換為 Python 代碼。

**主要功能：**
- Pine Script → Python 自動轉換
- 函數映射（SMA, EMA, RSI, MACD 等）
- 輸入參數提取
- 完整的輔助函數生成

### 2. indicator_framework.py
可擴展的指標框架，用於創建、管理和可視化技術指標。

**主要類：**
- `BaseIndicator`: 所有指標的基類
- `IndicatorManager`: 指標管理器
- 預置指標：
  - `MovingAverageIndicator`: 移動平均線
  - `RSIIndicator`: 相對強度指數
  - `MACDIndicator`: MACD
  - `BollingerBandsIndicator`: 布林帶

### 3. model_ensemble_gui_v2.py
整合的圖形界面，包含：
- 數據加載
- 指標轉換器
- Smart Money Concepts 分析
- 自定義指標管理和可視化

---

## 使用流程

### 第一步：準備指標源碼

獲取你想要的交易指標源碼。支持的格式：

#### Pine Script 示例
```pinescript
//@version=5
indicator(title="Custom Momentum", overlay=false)

length = input(14, "RSI Length")
source = input(close, "Source")

rsi_value = ta.rsi(source, length)

plot(rsi_value, color=color.blue, linewidth=2)
plot(70, color=color.red, linewidth=1)
plot(30, color=color.green, linewidth=1)

alertcondition(rsi_value > 70, title="Overbought")
alertcondition(rsi_value < 30, title="Oversold")
```

### 第二步：在轉換器中轉換

1. 運行 `model_ensemble_gui_v2.py`
2. 切換到 "Indicator Converter" 標籤頁
3. 選擇指標類型（Pine Script 或 Custom Python）
4. 將原始碼貼入左側輸入框
5. 點擊 "Convert Code" 按鈕
6. 轉換後的 Python 代碼將出現在右側

### 第三步：自定義轉換後的代碼

雖然轉換器會自動進行基本的語法轉換，但對於複雜的邏輯，你可能需要手動調整：

```python
from indicator_framework import BaseIndicator
import pandas as pd
import numpy as np

class CustomMomentumIndicator(BaseIndicator):
    def __init__(self):
        super().__init__(
            name='Custom Momentum',
            description='自定義動量指標',
            overlay=False  # False: 單獨圖表, True: 疊加在K線上
        )
        self.parameters = {
            'rsi_length': 14,
            'overbought': 70,
            'oversold': 30,
        }
    
    def calculate(self, df: pd.DataFrame):
        """計算指標"""
        close = df['close']
        
        # 計算 RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=self.parameters['rsi_length']
        ).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(
            window=self.parameters['rsi_length']
        ).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 添加主線
        self.add_plot('RSI', rsi, color='#1E90FF', width=2.0)
        
        # 添加水平線
        self.add_plot(
            'Overbought',
            pd.Series([self.parameters['overbought']] * len(rsi), index=rsi.index),
            color='#FF0000', line_style='--', width=1.0, alpha=0.7
        )
        self.add_plot(
            'Oversold',
            pd.Series([self.parameters['oversold']] * len(rsi), index=rsi.index),
            color='#00FF00', line_style='--', width=1.0, alpha=0.7
        )
        
        # 添加信號
        for i in range(1, len(rsi)):
            if not pd.isna(rsi.iloc[i]):
                if rsi.iloc[i] > self.parameters['overbought']:
                    self.add_signal('overbought', i, rsi.iloc[i])
                elif rsi.iloc[i] < self.parameters['oversold']:
                    self.add_signal('oversold', i, rsi.iloc[i])
        
        return {
            'plots': self.plots,
            'signals': self.signals,
        }
```

### 第四步：集成到應用

#### 選項 A: 在 GUI 中動態添加

將轉換後的代碼保存為 Python 文件（例如 `custom_momentum.py`），然後修改 GUI 代碼：

```python
# 在 model_ensemble_gui_v2.py 中
from custom_momentum import CustomMomentumIndicator

def add_custom_indicator(self):
    indicator = CustomMomentumIndicator()
    self.indicator_manager.register_indicator(indicator)
    self.indicator_listbox.insert(tk.END, indicator.name)
```

#### 選項 B: 直接在框架中調用

```python
from indicator_framework import IndicatorManager
from custom_momentum import CustomMomentumIndicator

# 初始化管理器
manager = IndicatorManager()

# 註冊指標
manager.register_indicator(CustomMomentumIndicator())

# 加載數據
df = pd.read_parquet('data/btc_15m.parquet')

# 計算所有指標
results = manager.calculate_all(df)

# 繪製
fig = manager.plot_indicators(df)
plt.show()
```

---

## 常見轉換場景

### 場景 1: 簡單的移動平均線

**Pine Script:**
```pinescript
fastMA = ta.sma(close, 20)
slowMA = ta.sma(close, 50)
plot(fastMA, color=color.blue)
plot(slowMA, color=color.red)
```

**轉換後 Python:**
```python
def calculate(self, df: pd.DataFrame):
    close = df['close']
    fast_ma = close.rolling(window=20).mean()
    slow_ma = close.rolling(window=50).mean()
    
    self.add_plot('Fast MA', fast_ma, color='#0000FF')
    self.add_plot('Slow MA', slow_ma, color='#FF0000')
    
    return {'plots': self.plots, 'signals': self.signals}
```

### 場景 2: 帶信號的振蕩器

**Pine Script:**
```pinescript
rsi = ta.rsi(close, 14)
plot(rsi, color=color.blue)
hline(70, color=color.red)
hline(30, color=color.green)
alertcondition(rsi > 70, title="Overbought")
alertcondition(rsi < 30, title="Oversold")
```

**轉換後 Python:**
```python
def calculate(self, df: pd.DataFrame):
    close = df['close']
    
    # 計算 RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    
    self.add_plot('RSI', rsi, color='#0000FF')
    
    # 添加信號
    for i in range(len(rsi)):
        if rsi.iloc[i] > 70:
            self.add_signal('overbought', i, rsi.iloc[i])
        elif rsi.iloc[i] < 30:
            self.add_signal('oversold', i, rsi.iloc[i])
    
    return {'plots': self.plots, 'signals': self.signals}
```

---

## 在 GUI 中使用

### 加載數據
1. 打開 "Data Loading" 標籤
2. 點擊 "Load Default Data" 或 "Load Local CSV/Parquet"

### 轉換指標
1. 切換到 "Indicator Converter" 標籤
2. 選擇 "Pine Script"
3. 貼上源碼
4. 點擊 "Convert Code"
5. 複製輸出或保存為文件

### 使用自定義指標
1. 切換到 "Custom Indicators" 標籤
2. 點擊 "Add All Default" 添加預置指標
3. 點擊 "Plot Selected" 繪製

### 使用 SMC 分析
1. 切換到 "Smart Money Concepts" 標籤
2. 調整 "Swing Length" 參數
3. 點擊 "Analyze SMC"

---

## API 參考

### BaseIndicator 類

```python
class BaseIndicator(ABC):
    def __init__(self, name: str, description: str = '', overlay: bool = False)
    
    def set_parameter(self, param_name: str, value: Any)
    def add_plot(self, plot_name: str, data: pd.Series, color: str, 
                 line_style: str = '-', width: float = 1.5, alpha: float = 1.0)
    def add_signal(self, signal_type: str, index: int, value: float, 
                   signal_data: Dict = None)
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Dict[str, Any]
```

### IndicatorManager 類

```python
class IndicatorManager:
    def register_indicator(self, indicator: BaseIndicator)
    def calculate_all(self, df: pd.DataFrame) -> Dict[str, Any]
    def get_signals(self, indicator_name: str) -> List[Dict]
    def plot_indicators(self, df: pd.DataFrame, fig=None, max_subplots: int = 3)
```

---

## 故障排除

### 問題 1: 轉換後的代碼有語法錯誤
**解決方案：**
- 檢查原始代碼中的特殊 Pine Script 語法
- 手動調整複雜的邏輯
- 確保所有變量都有定義

### 問題 2: 指標不顯示
**解決方案：**
- 檢查 `add_plot()` 調用是否正確
- 確保 `calculate()` 方法返回 plots
- 驗證數據不為空

### 問題 3: 轉換器無法識別某些函數
**解決方案：**
- 手動添加函數映射
- 實現缺失的輔助函數
- 查看 `indicator_converter.py` 中的 `setup_mappings()` 方法

---

## 下一步

1. **擴展轉換器**：添加更多 Pine Script 函數的支持
2. **自動信號生成**：實現智能信號檢測
3. **指標組合**：支持多個指標的組合策略
4. **回測集成**：將指標信號與回測引擎集成

---

## 資源

- [Pine Script 文檔](https://www.tradingview.com/pine-script-docs/)
- [技術分析指標](https://en.wikipedia.org/wiki/Technical_analysis)
- [Python Pandas 文檔](https://pandas.pydata.org/docs/)

---

## 許可

MIT License
