# 模型優化策略：從「盲目預測方向」到「識別交易型態和聰明錢行為」

## 核心洞察

你現在提的想法，實際上是從「統計預測」轉向「結構識別」的範式轉變。這很重要，以下是為什麼：

### 1. 現在的問題本質

**現狀:**
- 模型在嘗試預測 15m BTC 的「下一根 K 線漲跌」
- 這基本上是在讓模型學習噪聲
- 31.83% 的準確度，其實接近「隨機 + 一點點噪聲擬合」

**為什麼會這樣？**
- 15m 級別的 K 線變動，太多是隨機性和市場微觀結構
- 真正可預測的 edge，通常不在「下一根會漲還是跌」
- 而在「這裡有 Supply/Demand Zone、有聰明錢跡象、有型態完成」這類**結構信息**

### 2. 你想法的正確性（為什麼有幫助）

#### A. 人類交易員為什麼能贏

一個好的人類交易員會做：
```
看圖 → 識別 Supply/Demand Zone
        ↓
    識別聰明錢跡象（例如 Imbalance、Liquidity grab）
        ↓
    看 K 線型態（例如 Inside Bar、Engulfing、反轉 pattern）
        ↓
    評估當前價格「在結構中的位置」
        ↓
    決策：概率期望正的交易
```

**關鍵點:** 他們不是預測「下根會漲或跌」，而是識別「這個位置有高概率反應」。

#### B. 機器學習能做什麼

機器學習應該被訓練來做：

1. **自動檢測結構（結構識別）**
   - Supply Zone: 前高點、賣壓集中區
   - Demand Zone: 前低點、買盤集中區
   - 聰明錢指標: Break of Structure (BOS)、Imbalance、Fair Value Gap
   - K 線型態: Engulfing、Inside Bar、Pin Bar、Double Top/Bottom

2. **在這些結構中定位當前價格（位置評估）**
   - 現在是接近 Supply 還是 Demand？
   - 最近是否發生 BOS（聰明錢方向變化）？
   - 型態是否完成

3. **基於結構預測短期反應（結構性預測）**
   - 而不是「盲目預測方向」
   - 而是「在這個已知結構中，反應機率」

### 3. 具體為什麼會更有效

#### 問題 1: 信噪比
```
現在做法: noise ratio ~ 90%
「下一根漲跌」基本上是白噪聲

新做法: noise ratio ~ 40-60%
「價格是否到達 Supply Zone」、「是否觸發聰明錢信號」
這些事件本身就比較「結構化」，不純是隨機
```

#### 問題 2: 標籤有意義性
```
現在的 y = (close_t+1 > close_t) ? 1 : 0
這個 y 沒有任何「主題」性質，純粹是價格漲跌

新的 y 可以是：
y = {
  1: 價格觸碰了 Supply Zone 並反彈
  0: 價格在 Demand Zone 或中性區
  2: 觸發了 Break of Structure（聰明錢方向改變）
}
這樣 y 本身就包含「人類交易員認為重要」的信息
```

#### 問題 3: 特徵更有 edge
```
現在的特徵: Technical indicators (RSI, MACD, Bollinger Band...)
問題: 這些本身也是推導出來的，信噪比還是低

新的特徵:
1. 結構性特徵:
   - 到最近 Supply Zone 的距離
   - 到最近 Demand Zone 的距離
   - 最近是否有 BOS（1 = 有新的 BOS, 0 = 沒有）
   - 距離上次反轉有多少根 K 線（momentum depletion signal）

2. 聰明錢指標特徵:
   - Imbalance 檢測（高低價之間是否有 gap）
   - Fair Value Gap (FVG) 檢測
   - 成交量分析（成交量是否異常）
   - Liquidity grab：最近是否以新高/新低快速反轉

3. 型態特徵:
   - 最近 5 根 K 線的型態 (1-hot encoding)
   - Inside Bar 檢測
   - Pin Bar 檢測
   - Engulfing 檢測
   - 型態完成度（0-1 之間）

這些特徵本身就具有「交易意義」，不是泛用的技術指標
```

---

## 實作路線圖

### Phase 1: 建立結構檢測模組（1-2 週）

**目標:** 構建能自動檢測 Supply/Demand Zone、聰明錢信號、K 線型態的模組

#### 1.1 Supply/Demand Zone 檢測
```python
class SupplyDemandDetector:
    def detect_zones(self, df, lookback=100):
        """
        檢測最近的 Supply (前高點、賣壓區) 和 Demand (前低點、買盤區)
        
        Returns:
        {
            'supply_zones': [(price, strength, timestamp), ...],
            'demand_zones': [(price, strength, timestamp), ...],
            'nearest_supply': price,
            'nearest_demand': price,
            'distance_to_supply': distance_pct,
            'distance_to_demand': distance_pct
        }
        """
```

**核心邏輯:**
- Supply Zone: 前高點、高點附近有明顯賣壓（例如成交量尖峰）
- Demand Zone: 前低點、低點附近有明顯買盤
- 強度計分: 根據「被測試多少次」+ 「時間衰減」

#### 1.2 聰明錢指標檢測
```python
class SmartMoneyDetector:
    def detect_bos(self, df):
        """Break of Structure: 前低點被打破（向上 BOS）或前高點被打破（向下 BOS）"""
    
    def detect_imbalance(self, df):
        """高低價之間有 gap，未被 fill（聰明錢遺留下的不平衡）"""
    
    def detect_fvg(self, df):
        """Fair Value Gap: 連續 3 根 K 線中的 gap，市場後續傾向回來 fill"""
    
    def detect_liquidity_grab(self, df):
        """極速觸及新高/新低，然後快速反轉，代表聰明錢掃單後轉向"""
```

#### 1.3 K 線型態檢測
```python
class CandlestickPatternDetector:
    def detect_inside_bar(self, df):
        """當前 K 線完全在前一根 K 線高低之內"""
    
    def detect_pin_bar(self, df):
        """長引線 K 線，代表反轉 signal"""
    
    def detect_engulfing(self, df):
        """吞沒型態"""
    
    def detect_pattern_completion(self, df, pattern_type='double_top'):
        """檢測 Double Top/Bottom 等更複雜的型態"""
```

---

### Phase 2: 特徵工程（1-2 週）

**目標:** 把檢測結果轉換成機器學習可用的特徵

#### 2.1 結構性特徵
```python
class StructureFeatures:
    def compute_features(self, df):
        features = {
            # 距離特徵
            'pct_to_nearest_supply': 0.05,  # 當前價格距最近 Supply Zone 的 %
            'pct_to_nearest_demand': -0.02,
            
            # 強度特徵
            'nearest_supply_strength': 0.8,  # 供應區強度 (0-1)
            'nearest_demand_strength': 0.6,
            
            # 結構狀態
            'recent_bos_direction': 1,  # 1 = 向上 BOS, -1 = 向下 BOS, 0 = 無
            'bars_since_bos': 5,  # 距離最近 BOS 有多少根 K 線
            
            # 型態信息
            'current_pattern': 'inside_bar',  # 當前 K 線型態
            'pattern_completion_pct': 0.85,  # 型態完成度
            
            # 聰明錢信號
            'imbalance_detected': 1,  # 是否有 imbalance
            'fvg_above': 0.8,  # 上方 FVG 距離 (%)
            'fvg_below': -0.5,
            'liquidity_grab_signal': 0,  # 是否發生 liquidity grab
        }
        return features
```

#### 2.2 結合技術指標
```python
# 原有的 RSI, MACD 等指標仍然保留
# 但現在作為「次要信息」，主要信息是結構性特徵

features_combined = {
    # 結構特徵（主要）: 60%
    'structure_features': {...},
    
    # 聰明錢特徵（主要）: 25%
    'smart_money_features': {...},
    
    # 技術指標特徵（輔助）: 15%
    'technical_features': {
        'rsi': 0.65,
        'macd': 0.02,
        ...
    }
}
```

---

### Phase 3: 標籤重新設計（1 週）

**目標:** 設計有意義的標籤，而不是盲目「下根漲跌」

#### 3.1 多層級標籤設計
```python
def create_smart_labels(df, lookback=32):
    """
    在未來 lookback 根 K 線內，檢測是否發生了交易員會在意的事件
    
    Return:
    {
        'label': 0/1/2,  # 0=無邊界反應, 1=供應區反應, 2=需求區反應
        'confidence': 0.7,  # 該事件發生的確定性
        'reward': 0.015,  # 如果在該區域反應，的獲利幅度
    }
    """
    
    # 檢測：未來是否會觸及 Supply 並反彈
    future_max = df['high'].rolling(lookback).max()
    future_min = df['low'].rolling(lookback).min()
    
    supply_zones = detect_supply_zones(df)
    demand_zones = detect_demand_zones(df)
    
    labels = []
    for i in range(len(df) - lookback):
        current_price = df.iloc[i]['close']
        future_prices = df.iloc[i:i+lookback]
        
        # 檢測是否觸及 Supply 並反彈
        for sz_price, sz_strength in supply_zones:
            if any(future_prices['high'] >= sz_price):
                # 觸及了 Supply
                touch_idx = list(future_prices['high'] >= sz_price).index(True)
                after_touch = future_prices.iloc[touch_idx:]
                
                # 檢查是否反彈
                if len(after_touch) > 1 and after_touch['low'].min() < sz_price * 0.995:
                    labels.append({
                        'label': 1,  # Supply Zone 反應
                        'confidence': sz_strength,
                        'reward': (after_touch['high'].max() - sz_price) / sz_price
                    })
                    break
        else:
            # 檢測 Demand Zone
            for dz_price, dz_strength in demand_zones:
                if any(future_prices['low'] <= dz_price):
                    touch_idx = list(future_prices['low'] <= dz_price).index(True)
                    after_touch = future_prices.iloc[touch_idx:]
                    
                    if len(after_touch) > 1 and after_touch['high'].max() > dz_price * 1.005:
                        labels.append({
                            'label': 2,  # Demand Zone 反應
                            'confidence': dz_strength,
                            'reward': (dz_price - after_touch['low'].min()) / dz_price
                        })
                        break
            else:
                labels.append({'label': 0, 'confidence': 0.5, 'reward': 0})
    
    return labels
```

#### 3.2 樣本篩選
```python
# 不是所有樣本都用於訓練
# 只用「高質量」樣本：
# - 明確接觸到結構的樣本（label != 0）
# - 信心度 > 0.6 的樣本
# - 預期報酬 > 0.5% 的樣本

quality_samples = [s for s in labels if s['confidence'] > 0.6 and s['reward'] > 0.005]
training_data = df[len(quality_samples) > 0]
```

---

### Phase 4: 模型架構調整（1-2 週）

**目標:** 改造現有 LightGBM + CatBoost 架構，以適應新的特徵和標籤

#### 4.1 特徵重要性調整
```python
# LightGBM 參數
lgb_params = {
    'n_estimators': 500,  # 可以更多，因為信號品質更好
    'max_depth': 15,  # 結構性特徵需要更深的樹
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,  # 可以降低，因為樣本更有質量
    'objective': 'binary',
    'metric': 'auc',  # 改用 AUC 而非準確度
    'scale_pos_weight': len(neg_samples) / len(pos_samples),  # 處理不平衡
}
```

#### 4.2 多輸出頭
```python
# 不是只輸出「方向」
# 而是輸出多個量度：

model_output = {
    'supply_zone_touch_prob': 0.72,  # 未來會觸及 Supply 並反彈的機率
    'demand_zone_touch_prob': 0.18,  # 未來會觸及 Demand 並反彈的機率
    'neutral_prob': 0.10,  # 都不會發生
    'expected_reward': 0.012,  # 期望報酬 %
    'confidence': 0.75,  # 整體信心度
}
```

---

### Phase 5: 回測與評估（1-2 週）

#### 5.1 評估指標（不再用準確度）
```python
# 舊方式
Accuracy: 31.83%  # 沒意義

# 新方式
Precision@Supply: 0.62  # 在 Supply Zone 反應的信號，有 62% 確實反彈
Recall@Supply: 0.48  # 所有實際 Supply Zone 反應，捉到 48%
ROC-AUC: 0.71  # 區分「會反彈」和「不會」的能力
Sharpe Ratio: 1.8  # 基於該策略的風調後報酬
Win Rate: 58%  # 在高信心信號上的交易成功率
Profit Factor: 2.3  # 獲利交易總額 / 虧損交易總額
```

#### 5.2 策略回測框架
```python
class StructureBasedStrategy:
    def generate_signals(self, model_output):
        """
        基於模型輸出生成交易信號
        
        不是「概率 > 0.5 就做多」
        而是「高信心 Supply Zone 信號 + 確認 K 線型態」才做空
        """
        
        signals = []
        
        # 供應區做空
        if (model_output['supply_zone_touch_prob'] > 0.6 and 
            model_output['confidence'] > 0.7 and
            is_reversal_pattern()):  # 附加 K 線型態確認
            signals.append({
                'direction': -1,  # 做空
                'entry_type': 'supply_zone',
                'risk_ratio': model_output['expected_reward'] / stop_loss_pct,
                'confidence': model_output['confidence']
            })
        
        # 需求區做多
        if (model_output['demand_zone_touch_prob'] > 0.6 and 
            model_output['confidence'] > 0.7 and
            is_reversal_pattern()):
            signals.append({
                'direction': 1,  # 做多
                'entry_type': 'demand_zone',
                'risk_ratio': model_output['expected_reward'] / stop_loss_pct,
                'confidence': model_output['confidence']
            })
        
        return signals
    
    def calculate_return(self, signals, actual_price_movement):
        """計算實際報酬"""
```

---

## 預期改善

### 現狀 vs 新方案

| 指標 | 現狀 | 預期新方案 | 改善 |
|------|------|---------|------|
| 整體準確度 | 31.83% | N/A* | - |
| Precision@High Conf | 不適用 | ~65-70% | +3x |
| ROC-AUC | ~0.5 | ~0.70-0.75 | +40% |
| 高信心信號比例 | 0.4% | 5-10% | +10x |
| 高信心信號準確度 | 6.37% | ~60-65% | +10x |
| Sharpe Ratio | 負 | 1.5-2.5 | 可交易 |
| 實際交易勝率 | N/A | 55-62% | 可用 |

*因為任務從「預測方向」改成「識別結構反應」，不再用準確度，而用 Precision/Recall

### 為什麼會改善

1. **任務變簡單了**
   - 不是「預測下根會漲還是跌」（噪聲 90%）
   - 而是「這個位置會不會反應」（噪聲 40%）

2. **標籤品質更好**
   - 現在的標籤：純噪聲
   - 新的標籤：包含交易邏輯的事件

3. **特徵有交易意義**
   - 現在的特徵：泛用技術指標
   - 新的特徵：直接描述結構狀態

4. **模型學習的是規律，而不是噪聲**
   - 「價格接近 Supply Zone 時，傾向反彈」這是真實的市場現象
   - 「下一根 K 線會漲或跌」這是隨機的

---

## 實作建議

### 優先順序

1. **先做 Supply/Demand Detection**（最有 impact）
   - 這是整個方向的基礎
   - 實作起來也最直接

2. **加上聰明錢指標檢測**（增加 edge）
   - BOS, Imbalance, FVG
   - 這些東西人類交易員看重的，模型也應該看到

3. **K 線型態檢測**（確認信號）
   - 用來確認 Supply/Demand 反應
   - 而不是獨立預測

4. **改造標籤系統**（3-6 個月回測驗證）
   - 新標籤應該基於前面 3 項的檢測結果
   - 逐步驗證，看實際交易報酬

5. **微調模型架構**（1-2 個月）
   - 基於新特徵和標籤調參
   - 重點是 AUC / Precision，而非準確度

---

## 核心哲學

**從「讓模型猜測隨機事件」轉向「讓模型識別結構性機會」**

你的直覺完全正確。機器學習應該被用來：

1. 檢測人眼可能漏掉的結構細節
2. 在龐大的歷史數據中尋找結構性事件的回報模式
3. 評估每個事件發生的機率和預期報酬
4. 組合成可交易的信號

而**不應該**被用來：

1. 預測隨機的價格變動
2. 猜測下一根 K 線的顏色
3. 學習沒有交易邏輯的統計關係

---

## 下一步

如果你同意這個方向，我們可以：

1. **寫第一版 `SupplyDemandDetector`**（1-2 天）
   - 在你的 BTC 15m 數據上測試
   - 看看能檢測到多少合理的 Zone

2. **寫 `SmartMoneyDetector` 模組**（2-3 天）
   - BOS, Imbalance, FVG, Liquidity Grab
   - 驗證這些信號在歷史上的真實反應率

3. **構建新的特徵集和標籤系統**（1 週）
   - 用新特徵和標籤訓練一版模型
   - 和現有模型對比

你想從哪裡開始？我可以直接給你程式碼，推到 repo。
