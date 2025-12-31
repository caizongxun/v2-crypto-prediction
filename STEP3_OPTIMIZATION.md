# 第三步：黃金公式參數優化

## 目標

使用演算法（Grid Search / Random Search / Bayesian Optimization）在 BTC 15m 歷史數據上尋找最優的參數配置。

## 優化流程

### 階段 1：粗糙搜尋 (Grid Search)

**目的**：快速找到大致的參數範圍

**優點**：
- 簡單易懂
- 覆蓋範圍完整
- 並行計算友好

**缺點**：
- 計算量大 (3^8 = 6561 種組合)
- 慢速

**實施**：
```bash
python optimize_formula.py
```

**預期結果**：
- 試驗次數：6561 次
- 耗時：數小時
- 輸出：Grid Search 最佳參數 (`results/step1_grid_search.json`)

### 階段 2：精細搜尋 (Random Search)

**目的**：在參數空間中隨機取樣，高效探索

**優點**：
- 快速 (200 次試驗 ~ 10 分鐘)
- 能發現 Grid 遺漏的最優值
- 易於調整試驗次數

**缺點**：
- 不夠系統
- 可能遺漏某些區域

**實施**：自動包含在 `optimize_formula.py` 中

**預期結果**：
- 試驗次數：200 次
- 耗時：~10 分鐘
- 輸出：Random Search 最佳參數 (`results/step2_random_search.json`)

### 階段 3：智慧搜尋 (Bayesian Optimization with Optuna)

**目的**：使用機器學習自適應地尋找最優參數

**優點**：
- 最高效 (100 次試驗就能找到優秀解)
- 自適應學習
- 最少試驗次數

**缺點**：
- 需要安裝 Optuna (`pip install optuna`)
- 複雜度高

**實施**：自動包含在 `optimize_formula.py` 中

**預期結果**：
- 試驗次數：100 次
- 耗時：~10 分鐘
- 輸出：Bayesian 最佳參數 (`results/step3_bayesian_search.json`)

---

## 優化目標函數

### 目前選擇：Sharpe Ratio

```python
Sharpe Ratio = (平均收益 - 無風險利率) / 標準差
```

**為什麼選 Sharpe**：
- 同時考慮收益和風險
- 不會被單次大賺引導
- 風險調整後的衡量指標

### 其他可選目標

#### 1. 勝率 (Win Rate)
```python
Win Rate = 勝利次數 / 總交易次數
```
- 簡單直觀
- 可能導致過度優化

#### 2. 收益率 (Return)
```python
Return = (結束資金 - 開始資金) / 開始資金
```
- 考慮絕對收益
- 忽略風險

#### 3. 利潤因子 (Profit Factor)
```python
Profit Factor = 總利潤 / 總虧損
```
- 衡量風險/報酬比
- > 2.0 認為不錯

#### 4. 最大回撤 (Max Drawdown)
```python
Max Drawdown = (峰值 - 谷值) / 峰值
```
- 衡量最大虧損
- 風險指標

---

## 要優化的參數

### 指標參數 (15 個)

| 參數 | 含義 | 範圍 | 優先級 |
|------|------|------|--------|
| `fast_ema` | 短期 EMA | 8-25 | ⭐⭐⭐ |
| `slow_ema` | 長期 EMA | 30-100 | ⭐⭐⭐ |
| `supertrend_period` | SuperTrend 週期 | 8-15 | ⭐⭐ |
| `supertrend_multiplier` | SuperTrend 倍數 | 1.5-5.0 | ⭐⭐ |
| `adx_threshold` | ADX 最低閾值 | 20-30 | ⭐⭐ |
| `rsi_period` | RSI 週期 | 10-20 | ⭐⭐⭐ |
| `rsi_oversold` | RSI 超賣線 | 20-40 | ⭐⭐ |
| `rsi_overbought` | RSI 超買線 | 60-80 | ⭐⭐ |
| `roc_period` | ROC 週期 | 8-20 | ⭐⭐ |
| `volume_spike` | 成交量峰值倍數 | 1.2-2.5 | ⭐ |
| `vwap_deviation` | VWAP 偏離 % | 0.5-2.0 | ⭐ |

### 權重參數 (4 個)

| 參數 | 含義 | 範圍 | 優先級 |
|------|------|------|--------|
| `trend_weight` | 趨勢權重 | 0.3-0.5 | ⭐⭐⭐ |
| `momentum_weight` | 動能權重 | 0.2-0.4 | ⭐⭐⭐ |
| `volume_weight` | 成交量權重 | 0.1-0.3 | ⭐⭐ |
| `confidence_threshold` | 信心閾值 | 0.55-0.75 | ⭐⭐⭐ |

---

## 執行步驟

### 快速版 (5 分鐘)

```bash
# 只執行 Random Search
python optimize_formula.py
```

### 完整版 (30 分鐘)

```bash
# 執行全部三個階段
python optimize_formula.py
```

### 自定義版

```python
from optimize_formula import optimize_step_2_random_search
from data import load_btc_data

df = load_btc_data(hf_token="...", start_date='2024-01-01')
result = optimize_step_2_random_search(df)
```

---

## 結果解讀

### 輸出文件

```
results/
├── step1_grid_search.json      # Grid Search 結果
├── step2_random_search.json    # Random Search 結果
└── step3_bayesian_search.json  # Bayesian 結果
```

### JSON 結構

```json
{
  "search_method": "Random Search",
  "total_trials": 200,
  "duration_seconds": 600,
  "best_score": 1.2345,
  "best_params": {
    "fast_ema": 15,
    "slow_ema": 60,
    "rsi_period": 14,
    "...": "..."
  },
  "all_trials": [
    {
      "params": {...},
      "score": 0.8234
    }
  ]
}
```

### 評估指標

**Sharpe Ratio**
- < 0：虧損策略
- 0-1：不夠好
- 1-2：合理
- 2-3：很好
- > 3：優秀

**勝率**
- > 50%：正期望值
- 40-50%：需要大盈虧比
- < 40%：差

**利潤因子**
- > 2.0：很好
- 1.5-2.0：合理
- < 1.5：差

---

## 注意事項

### 1. 過度擬合 (Overfitting)

**風險**：
- 參數在歷史數據上表現完美，但在未來失效
- 優化了噪聲而非信號

**防護**：
- 使用 walk-forward validation（滑動時間窗口）
- 在不同時間段驗證
- 只優化高優先級參數

### 2. 樣本外驗證

```python
# 用 2024 數據優化
# 用 2025 數據驗證（未來數據）
```

### 3. 參數敏感性分析

查看 JSON 文件中的 `all_trials`：
- 找到分數最高的 20 個試驗
- 分析共同特徵
- 識別穩健的參數範圍

---

## 優化建議

### 階段 1：粗糙優化

1. 先只優化 **最關鍵的 3 個參數**：
   - `fast_ema`
   - `slow_ema`
   - `trend_weight`

2. 使用 Grid Search：
   ```python
   param_grid = {
       'fast_ema': [12, 15, 18],
       'slow_ema': [50, 60, 70],
       'trend_weight': [0.35, 0.4, 0.45],
       # 其他用默認值
   }
   ```

### 階段 2：精細優化

1. 在第一階段最優值周圍進行 Random Search
2. 逐步加入更多參數
3. 使用較大的試驗數（200-500）

### 階段 3：最終驗證

1. 用 Bayesian Optimization 驗證
2. 在樣本外數據進行回測
3. 計算統計信度（比如 Sharpe > 1.0）

---

## 下一步

第四步會進行：
1. **Walk-Forward Validation** - 滑動時間窗口驗證
2. **敏感性分析** - 參數變化對結果的影響
3. **統計檢驗** - 驗證結果的統計顯著性
4. **實盤模擬** - 模擬真實交易成本和滑點

---

## 參考資源

- Grid Search：簡單，覆蓋全
- Random Search：快速，發現意外
- Bayesian Optimization：最優效，聰慧搜尋

**推薦流程**：Grid → Random → Bayesian
