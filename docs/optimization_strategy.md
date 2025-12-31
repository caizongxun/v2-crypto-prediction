# 模型優化方案：51.79% → 70%+

## 📊 當前分析

目前方向準確率: 51.79% (略優於隨機 50%)
目標準確率: 70%+
改善空間: +18.21%

## 核心優化策略

### Strategy 1: 特徵工程升級 ⭐⭐⭐⭐⭐

**新增特徵:**

1. **滯後特徵** (5 個)
   - close_lag1, close_lag2, close_pct_lag1
   - direction_lag1, direction_lag2
   - 預計提升: +2-3%

2. **動量特徵** (4 個)
   - rsi_momentum, rsi_acceleration
   - macd_momentum, bb_squeeze
   - 預計提升: +1-2%

3. **市場結構特徵** (4 個)
   - price_high_ratio, price_low_ratio
   - consecutive_up, consecutive_down
   - 預計提升: +1-2%

4. **相對波動率特徵** (2 個)
   - atr_ratio_to_mean, volume_ratio_to_mean
   - 預計提升: +0.5-1%

**總計: 14 → 36 個特徵，預計 +5-8% 準確率**

### Strategy 2: 模型算法升級 ⭐⭐⭐⭐

**升級路徑:**
- RandomForest → LightGBM (更適合梯度提升)
- 預計提升: +3-5%

**可選集合投票:**
- 組合 LightGBM + XGBoost + RandomForest
- 預計提升: +2-3%

### Strategy 3: 特徵選擇優化 ⭐⭐⭐

- 使用 Permutation Importance 選擇重要特徵
- 移除低貢獻度特徵
- 預計提升: +1-2%

### Strategy 4: 數據擴展 ⭐⭐⭐⭐

- 訓練數據: 6 個月 → 2 年 (80,000+ 樣本)
- 包含更多市場週期
- 預計提升: +2-3%

## 優化路線圖

```
基線: 51.79%
├─ Phase 1 (特徵工程) +6% → 57.79%
├─ Phase 2 (模型升級) +4% → 61.79%
├─ Phase 3 (特徵選擇) +2% → 63.79%
└─ Phase 4 (數據擴展) +3% → 70%+
```

## 實施優先級

**必做:**
1. 添加滯後特徵
2. 升級到 LightGBM

**強烈推薦:**
3. 擴展訓練數據到 2 年
4. 添加市場結構特徵

**可選:**
5. 集合投票
6. 特徵選擇優化
