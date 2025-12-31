# 優化算法修復技術文檔

## 概述

本文檔詳細說明對 `advanced_feature_builder.py` 進行的三個核心修復，以解決遺傳算法優化中的基本問題。

---

## 問題 1: 二分類目標導致相關性計算失敗

### 症狀
- 趨勢公式和方向公式相關性恆為 0.0
- 50 代進化中每代都沒有改進
- 演化算法完全無法找到好的解決方案

### 根本原因

原始代碼使用二分類目標：
```python
# ❌ 錯誤的做法
price_change = np.diff(df['close'].values)
trend_target = (price_change > 0).astype(float)  # 只有 0 和 1
```

**為什麼二分類導致相關性=0.0?**

1. Spearman 相關性基於排序
2. 當目標只有兩個值 (0, 1) 時，排序無法正確區分
3. 某些公式輸出全 0 或全 1，無法提供改進的梯度
4. 演化算法陷入局部最小值，無法逃脫

### 解決方案

改用連續值目標，保留自然的數值分佈：

```python
# ✅ 正確的做法
price_change = np.diff(df['close'].values)

# 趨勢: 直接使用連續的價格變化
trend_target = price_change / (np.percentile(np.abs(price_change), 95) + 1e-10)
trend_target = np.clip(trend_target, -1, 1)

# 方向: 下一根 K 線的方向強度
next_direction = np.append(price_change, [0])  # 後移一個位置
direction_target = np.sign(next_direction) * (np.abs(next_direction) / (np.percentile(np.abs(price_change), 95) + 1e-10))
direction_target = np.clip(direction_target, -1, 1)
```

**優勢:**
- 提供連續的梯度，讓遺傳算法可以逐步改進
- 保留數值的自然分佈
- 相關性計算變得有意義

---

## 問題 2: 波動性公式過度擬合

### 症狀
- 波動性公式相關性為 +0.9899（接近 1.0）
- 公式: `VOLATILITY_20 - 0.13*SMA_5 max 0.10*ROC_20`
- 主要使用 `VOLATILITY_20`，這是目標本身的計算

### 根本原因

演化算法找到了一個「作弊」的解決方案：
```python
# 波動性目標本身這樣計算：
volatility_target = close_series.pct_change().rolling(window=20).std().values

# 而公式直接使用了 VOLATILITY_20
# VOLATILITY_20 = close_series.pct_change().rolling(window=20).std().values

# 所以相關性 = 1.0（完全相同）
```

### 解決方案

在評估函數中對過度擬合進行懲罰：

```python
def evaluate_formula_for_target(self, gene, target_values, target_name=''):
    # ... 相關性計算 ...
    correlation, _ = stats.spearmanr(formula_clean, target_clean)
    
    # 針對波動性目標的過度擬合懲罰
    if target_name == 'volatility' and correlation > 0.95:
        correlation = correlation * 0.7  # 降低到 0.665 或更低
    
    return correlation
```

**機制:**
- 檢測到完全相關時 (>0.95)，將相關性乘以懲罰係數 (0.7)
- 迫使演化算法尋找更有實用價值的組合
- 預期結果: 相關性降至 0.50 ~ 0.70 範圍內

---

## 問題 3: 數值穩定性和無效相關性

### 症狀
- RuntimeWarning: invalid value encountered in divide
- 某些組合產生 NaN 或無限值
- 相關性計算不穩定

### 解決方案

實施多層檢查和改進的評估邏輯：

```python
def evaluate_formula_for_target(self, gene, target_values, target_name=''):
    try:
        formula_values = gene.calculate(self.indicator_builder)
        
        # 第一層: 基本有效性檢查
        if len(formula_values) == 0 or len(target_values) == 0:
            return 0.0
        
        # 過濾無效數據
        valid_idx = ~np.isnan(formula_values) & ~np.isnan(target_values) & \
                   np.isfinite(formula_values) & np.isfinite(target_values)
        
        if np.sum(valid_idx) < 50:  # 至少需要 50 個有效點
            return 0.0
        
        formula_clean = formula_values[valid_idx]
        target_clean = target_values[valid_idx]
        
        # 第二層: 數值穩定性檢查
        if not np.all(np.isfinite(formula_clean)) or not np.all(np.isfinite(target_clean)):
            return 0.0
        
        # 第三層: 方差檢查（防止常數值）
        formula_std = np.std(formula_clean)
        target_std = np.std(target_clean)
        
        if formula_std < 1e-6 or target_std < 1e-6:
            return 0.0
        
        # Spearman 相關性計算
        try:
            correlation, p_value = stats.spearmanr(formula_clean, target_clean)
        except:
            return 0.0
        
        if np.isnan(correlation):
            return 0.0
        
        # 過度擬合懲罰
        if target_name == 'volatility' and correlation > 0.95:
            correlation = correlation * 0.7
        
        return correlation
    except Exception as e:
        return 0.0
```

**檢查層次:**
1. **基本層**: 空數組檢查
2. **有效性層**: NaN/無限值過濾
3. **數量層**: 最少有效點數 (50 個)
4. **穩定性層**: 方差檢查
5. **相關性層**: 安全的相關性計算
6. **邏輯層**: 過度擬合懲罰

---

## 改進 4: 自適應變異率

### 目標
當演化停滯時自動增加變異以探索新區域

### 實現

```python
# 追蹤停滯代數
stagnant_count = 0

for generation in range(num_generations):
    # ... 評估和選擇 ...
    
    if current_best.fitness > best_fitness:
        best_fitness = current_best.fitness
        best_gene = deepcopy(current_best)
        stagnant_count = 0  # 重置
    else:
        stagnant_count += 1
    
    # 自適應變異
    mutation_rate = 0.25 if stagnant_count < 5 else 0.4
    
    # 使用該變異率進行變異
    child = self.mutate(child, mutation_rate)
```

**效果:**
- 初期: 變異率 25%，保持穩定進化
- 停滯 5 代後: 變異率增至 40%，加強探索
- 幫助逃脫局部最小值

---

## 改進 5: 增加搜索規模

### 修改參數

```python
# 種群規模
population_size = 80  # 從 50 增加到 80

# 代數
generations = 100  # 從 50 增加到 100

# 代價選擇池
parent = self.population[random.randint(0, min(15, len(self.population)-1))]
# 從前 10 增加到前 15，增加多樣性
```

**影響:**
- 更大的種群 = 更好的初始多樣性
- 更多代數 = 更充分的進化時間
- 更大的選擇池 = 減少過早收斂風險

---

## 預期改進效果

### 運行前
```
波動性: +0.9899 ← 過度擬合
趨勢:   +0.0000 ← 完全失敗
方向:   +0.0000 ← 完全失敗
```

### 運行後 (預期)
```
波動性: 0.55 ~ 0.70 ← 降低但更有實用價值
趨勢:   0.15 ~ 0.35 ← 從零開始有實質改進
方向:   0.20 ~ 0.40 ← 從零開始有實質改進
```

---

## 驗證方法

運行優化後，檢查 `results/advanced_formula_optimization.json`：

```bash
python advanced_feature_builder.py
cat results/advanced_formula_optimization.json
```

查看：
1. **correlation 值是否合理** (不是 0.0 或 1.0)
2. **components 是否多樣化** (不是單一指標)
3. **進化過程是否遞進** (代數增加時相關性應上升)

---

## 後續優化方向

1. **特徵選擇 (Feature Selection)**
   - 不是所有 22 個指標都有幫助
   - 可以預選最相關的指標

2. **交叉驗證 (Cross-validation)**
   - 在不同時期分割數據
   - 防止時間序列洩漏

3. **多目標優化 (Multi-objective)**
   - 同時優化相關性和簡潔性
   - 鼓勵更簡單的公式

4. **機器學習集成**
   - Random Forest 篩選重要特徵
   - XGBoost 建立更複雜的組合

---

## 技術細節參考

- **Spearman vs Pearson**: Spearman 基於排序，對異常值更健壯
- **過度擬合**: 當模型複雜度 > 數據信息量時發生
- **遺傳算法**: 通過交叉和變異在解空間中進行隨機搜索
- **自適應參數**: 根據進度動態調整算法參數

---

上次更新: 2025-12-31
