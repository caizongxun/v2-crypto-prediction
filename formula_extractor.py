#!/usr/bin/env python3
"""
3套特徵公式提取器

展示:
1. 波動性公式 - 完整代碼
2. 趨勢公式 - 完整代碼  
3. 方向公式 - 完整代碼

以及如何使用這些公式計算特徵
"""

import pandas as pd
import numpy as np
import json
import os


class VolatilityFormula:
    """
    波動性公式
    用途: 量化價格波動的大小 (0-1, 0=平穩, 1=劇烈)
    
    組成:
    - ATR (Average True Range) 權重 49.1%
    - Bollinger Bands 寬度 權重 36.7%
    - ROC 變化率 權重 24.4%
    """
    
    def __init__(self):
        self.atr_period = 20
        self.bb_period = 26
        self.bb_std = 1.807
        self.roc_period = 10  # 約5分鐘
        
        # 權重
        self.w_atr = 0.491
        self.w_bb = 0.367
        self.w_roc = 0.244
        
        # 正規化
        total = self.w_atr + self.w_bb + self.w_roc
        self.w_atr /= total
        self.w_bb /= total
        self.w_roc /= total
    
    def calculate(self, df: pd.DataFrame) -> np.ndarray:
        """
        計算波動性分數
        
        Args:
            df: DataFrame with columns ['high', 'low', 'close']
        
        Returns:
            volatility_score: [0, 1] array
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 1. ATR 計算
        tr = np.maximum(
            np.maximum(high - low, np.abs(high - np.roll(close, 1))),
            np.abs(low - np.roll(close, 1))
        )
        atr = pd.Series(tr).ewm(span=self.atr_period, adjust=False).mean().values
        atr_normalized = (atr / (close + 1e-10)) * 100
        atr_signal = np.clip(
            atr_normalized / np.percentile(atr_normalized[100:], 75),
            0, 1
        )
        
        # 2. Bollinger Bands 寬度
        sma = pd.Series(close).rolling(window=self.bb_period).mean().values
        std = pd.Series(close).rolling(window=self.bb_period).std().values
        bb_width = (2 * self.bb_std * std) / (sma + 1e-10)
        bb_signal = np.clip(
            bb_width / np.percentile(bb_width[100:], 75),
            0, 1
        )
        
        # 3. ROC (Rate of Change)
        roc = (close - np.roll(close, self.roc_period)) / \
              (np.roll(close, self.roc_period) + 1e-10) * 100
        roc_abs = np.abs(roc)
        roc_signal = np.clip(
            roc_abs / np.percentile(roc_abs[100:], 75),
            0, 1
        )
        
        # 加權組合
        volatility = (
            atr_signal * self.w_atr +
            bb_signal * self.w_bb +
            roc_signal * self.w_roc
        )
        
        return np.clip(volatility, 0, 1)
    
    def __repr__(self) -> str:
        return f"""
波動性公式 (VOLATILITY FORMULA)
{'='*60}
公式名: VOL(High, Low, Close)

數學表達式:
VOL = 0.49 × ATR_norm + 0.37 × BB_width + 0.24 × ROC_norm

其中:
  ATR_norm = ATR({self.atr_period}) / 75th_percentile(ATR)
  BB_width = 2 × {self.bb_std} × StdDev({self.bb_period}) / SMA({self.bb_period})
  ROC_norm = |ROC({self.roc_period})| / 75th_percentile(|ROC|)

輸出範圍: [0, 1]
  0   = 完全平穩 (波動極小)
  0.5 = 正常波動
  1   = 劇烈波動

應用: 用於量化市場波動性, 調整風險敞口
"""


class TrendFormula:
    """
    趨勢公式
    用途: 衡量趨勢的強度和方向 (0-1, 0=下跌趨勢, 1=上升趨勢)
    
    組成:
    - EMA 差異 (快速-緩慢) 權重 33.9%
    - MACD 動量 權重 20.7%
    - ADX 趨勢強度 權重 28.0%
    """
    
    def __init__(self):
        self.fast_ema = 22
        self.slow_ema = 50
        self.macd_signal = 9
        self.adx_period = 14
        
        # 權重
        self.w_ema = 0.339
        self.w_macd = 0.207
        self.w_adx = 0.280
        
        # 正規化
        total = self.w_ema + self.w_macd + self.w_adx
        self.w_ema /= total
        self.w_macd /= total
        self.w_adx /= total
    
    def calculate(self, df: pd.DataFrame) -> np.ndarray:
        """
        計算趨勢分數
        
        Args:
            df: DataFrame with columns ['high', 'low', 'close']
        
        Returns:
            trend_score: [0, 1] array (0=下跌, 0.5=無趨勢, 1=上升)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 1. EMA 差異
        close_series = pd.Series(close)
        fast_ema = close_series.ewm(span=self.fast_ema, adjust=False).mean().values
        slow_ema = close_series.ewm(span=self.slow_ema, adjust=False).mean().values
        ema_ratio = (fast_ema - slow_ema) / (slow_ema + 1e-10) * 100
        ema_signal = np.tanh(ema_ratio / 5)
        ema_signal = (ema_signal + 1) / 2  # 轉換到 [0, 1]
        
        # 2. MACD
        ema12 = close_series.ewm(span=12, adjust=False).mean().values
        ema26 = close_series.ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        signal_line = pd.Series(macd).ewm(span=self.macd_signal, adjust=False).mean().values
        macd_histogram = macd - signal_line
        macd_signal = np.tanh(macd_histogram / (np.std(macd_histogram) + 1e-10) / 5)
        macd_signal = (macd_signal + 1) / 2
        
        # 3. ADX (方向指數)
        delta = close_series.diff().values
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        plus_di = pd.Series(up).rolling(window=self.adx_period).sum().values / \
                  (pd.Series(np.abs(delta)).rolling(window=self.adx_period).sum().values + 1e-10) * 100
        minus_di = pd.Series(down).rolling(window=self.adx_period).sum().values / \
                   (pd.Series(np.abs(delta)).rolling(window=self.adx_period).sum().values + 1e-10) * 100
        
        di_sum = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_signal = np.clip(di_sum, 0, 1)
        
        # 加權組合
        trend = (
            ema_signal * self.w_ema +
            macd_signal * self.w_macd +
            adx_signal * self.w_adx
        )
        
        return np.clip(trend, 0, 1)
    
    def __repr__(self) -> str:
        return f"""
趨勢公式 (TREND FORMULA)
{'='*60}
公式名: TREND(High, Low, Close)

數學表達式:
TREND = 0.34 × EMA_signal + 0.21 × MACD_signal + 0.28 × ADX_signal

其中:
  EMA_signal = tanh((EMA({self.fast_ema}) - EMA({self.slow_ema})) / SMA({self.slow_ema}) × 100 / 5)
  MACD_signal = tanh((EMA12-26 - Signal({self.macd_signal})) / StdDev)
  ADX_signal = |+DI - -DI| / (+DI + -DI)

輸出範圍: [0, 1]
  0   = 強下跌趨勢
  0.5 = 無方向趨勢
  1   = 強上升趨勢

應用: 用於判斷趨勢方向和強度, 指導交易方向
"""


class DirectionFormula:
    """
    方向公式
    用途: 預測下一步價格方向的確定性 (0-1, 越接近0或1越確定)
    
    組成:
    - RSI 超買超賣 權重 21.7%
    - Stochastic 相對強度 權重 28.5%
    - ROC 速度方向 權重 36.0%
    """
    
    def __init__(self):
        self.rsi_period = 20
        self.stoch_k = 13
        self.stoch_d = 8
        self.roc_period = 10
        
        # 權重
        self.w_rsi = 0.217
        self.w_stoch = 0.285
        self.w_roc = 0.360
        
        # 正規化
        total = self.w_rsi + self.w_stoch + self.w_roc
        self.w_rsi /= total
        self.w_stoch /= total
        self.w_roc /= total
    
    def calculate(self, df: pd.DataFrame) -> np.ndarray:
        """
        計算方向分數
        
        Args:
            df: DataFrame with columns ['high', 'low', 'close']
        
        Returns:
            direction_score: [0, 1] array (0=看跌確定, 1=看漲確定)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 1. RSI
        close_series = pd.Series(close)
        delta = close_series.diff().values
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        avg_gain = gain_series.ewm(span=self.rsi_period, adjust=False).mean().values
        avg_loss = loss_series.ewm(span=self.rsi_period, adjust=False).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_signal = rsi / 100
        
        # 2. Stochastic
        low_min = close_series.rolling(window=self.stoch_k).min().values
        high_max = close_series.rolling(window=self.stoch_k).max().values
        
        stoch_k = (close - low_min) / (high_max - low_min + 1e-10) * 100
        stoch_k_series = pd.Series(stoch_k)
        stoch_d = stoch_k_series.rolling(window=self.stoch_d).mean().values
        stoch_signal = stoch_d / 100
        
        # 3. ROC
        roc = (close - np.roll(close, self.roc_period)) / \
              (np.roll(close, self.roc_period) + 1e-10) * 100
        roc_signal = np.tanh(roc / 10)
        roc_signal = (roc_signal + 1) / 2
        
        # 加權組合
        direction = (
            rsi_signal * self.w_rsi +
            stoch_signal * self.w_stoch +
            roc_signal * self.w_roc
        )
        
        return np.clip(direction, 0, 1)
    
    def __repr__(self) -> str:
        return f"""
方向公式 (DIRECTION FORMULA)
{'='*60}
公式名: DIRECTION(High, Low, Close)

數學表達式:
DIRECTION = 0.22 × RSI_signal + 0.29 × Stoch_signal + 0.36 × ROC_signal

其中:
  RSI_signal = RSI({self.rsi_period}) / 100
  Stoch_signal = StochD({self.stoch_k},{self.stoch_d}) / 100
  ROC_signal = tanh(ROC({self.roc_period}) / 10)

輸出範圍: [0, 1]
  0   = 確定看跌 (極可能下跌)
  0.5 = 中性 (漲跌難測)
  1   = 確定看漲 (極可能上漲)

應用: 用於判斷當前方向的確定性, 決定是否下單
"""


def main():
    print("\n" + "#" * 80)
    print("# 3套特徵公式展示")
    print("#" * 80)
    
    # 加載數據
    print("\n[一] 加載數據...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        print(f"✓ 加載成功: {len(df)} 根 K 線")
    except Exception as e:
        print(f"✗ 加載失敗: {e}")
        return
    
    # 創建公式
    vol_formula = VolatilityFormula()
    trend_formula = TrendFormula()
    direction_formula = DirectionFormula()
    
    # 打印公式
    print("\n" + "="*80)
    print(vol_formula)
    print("\n" + "="*80)
    print(trend_formula)
    print("\n" + "="*80)
    print(direction_formula)
    
    # 計算特徵
    print("\n[二] 計算特徵值...")
    vol_scores = vol_formula.calculate(df)
    trend_scores = trend_formula.calculate(df)
    direction_scores = direction_formula.calculate(df)
    
    print(f"✓ 波動性特徵計算完成")
    print(f"  平均值: {np.nanmean(vol_scores):.4f}")
    print(f"  範圍: [{np.nanmin(vol_scores):.4f}, {np.nanmax(vol_scores):.4f}]")
    
    print(f"\n✓ 趨勢特徵計算完成")
    print(f"  平均值: {np.nanmean(trend_scores):.4f}")
    print(f"  範圍: [{np.nanmin(trend_scores):.4f}, {np.nanmax(trend_scores):.4f}]")
    
    print(f"\n✓ 方向特徵計算完成")
    print(f"  平均值: {np.nanmean(direction_scores):.4f}")
    print(f"  範圍: [{np.nanmin(direction_scores):.4f}, {np.nanmax(direction_scores):.4f}]")
    
    # 保存特徵數據
    print("\n[三] 保存特徵數據...")
    features_df = pd.DataFrame({
        'timestamp': df.index,
        'price': df['close'].values,
        'volatility': vol_scores,
        'trend': trend_scores,
        'direction': direction_scores
    })
    
    os.makedirs('results', exist_ok=True)
    features_df.to_csv('results/extracted_features.csv', index=False)
    print(f"✓ 特徵已保存: results/extracted_features.csv")
    
    # 顯示樣本
    print("\n[四] 特徵樣本 (最後 5 根 K 線):")
    print(features_df.tail())
    
    # 保存公式代碼
    print("\n[五] 保存公式代碼...")
    
    formulas_code = f"""
# 3套最優化特徵提取公式
# 由遺傳算法自動優化
# 生成時間: 2024-12-31

import numpy as np
import pandas as pd

# ============================================================================
# 公式 1: 波動性公式
# ============================================================================

def volatility_formula(df):
    \"\"\"
    波動性公式 - 量化價格波動的大小
    
    輸入: DataFrame with columns ['high', 'low', 'close']
    輸出: [0, 1] 的波動性分數
      0   = 完全平穩
      0.5 = 正常波動
      1   = 劇烈波動
    \"\"\"
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # 參數
    atr_period = {vol_formula.atr_period}
    bb_period = {vol_formula.bb_period}
    bb_std = {vol_formula.bb_std:.3f}
    roc_period = {vol_formula.roc_period}
    
    # 1. ATR (Average True Range)
    tr = np.maximum(
        np.maximum(high - low, np.abs(high - np.roll(close, 1))),
        np.abs(low - np.roll(close, 1))
    )
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().values
    atr_norm = (atr / (close + 1e-10)) * 100
    atr_signal = np.clip(atr_norm / np.percentile(atr_norm[100:], 75), 0, 1)
    
    # 2. Bollinger Bands 寬度
    sma = pd.Series(close).rolling(window=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period).std().values
    bb_width = (2 * bb_std * std) / (sma + 1e-10)
    bb_signal = np.clip(bb_width / np.percentile(bb_width[100:], 75), 0, 1)
    
    # 3. ROC (Rate of Change)
    roc = np.abs((close - np.roll(close, roc_period)) / \
                 (np.roll(close, roc_period) + 1e-10) * 100)
    roc_signal = np.clip(roc / np.percentile(roc[100:], 75), 0, 1)
    
    # 加權組合
    w_atr = {vol_formula.w_atr:.4f}
    w_bb = {vol_formula.w_bb:.4f}
    w_roc = {vol_formula.w_roc:.4f}
    
    volatility = w_atr * atr_signal + w_bb * bb_signal + w_roc * roc_signal
    return np.clip(volatility, 0, 1)


# ============================================================================
# 公式 2: 趨勢公式
# ============================================================================

def trend_formula(df):
    \"\"\"
    趨勢公式 - 衡量上升/下跌趨勢的強度
    
    輸入: DataFrame with columns ['high', 'low', 'close']
    輸出: [0, 1] 的趨勢分數
      0   = 強下跌趨勢
      0.5 = 無明確趨勢
      1   = 強上升趨勢
    \"\"\"
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    close_series = pd.Series(close)
    
    # 參數
    fast_ema = {trend_formula.fast_ema}
    slow_ema = {trend_formula.slow_ema}
    macd_signal_period = {trend_formula.macd_signal}
    adx_period = {trend_formula.adx_period}
    
    # 1. EMA 差異
    fast_ema_val = close_series.ewm(span=fast_ema, adjust=False).mean().values
    slow_ema_val = close_series.ewm(span=slow_ema, adjust=False).mean().values
    ema_ratio = (fast_ema_val - slow_ema_val) / (slow_ema_val + 1e-10) * 100
    ema_signal = np.tanh(ema_ratio / 5)
    ema_signal = (ema_signal + 1) / 2
    
    # 2. MACD
    ema12 = close_series.ewm(span=12, adjust=False).mean().values
    ema26 = close_series.ewm(span=26, adjust=False).mean().values
    macd = ema12 - ema26
    signal_line = pd.Series(macd).ewm(span=macd_signal_period, adjust=False).mean().values
    macd_histogram = macd - signal_line
    macd_signal = np.tanh(macd_histogram / (np.std(macd_histogram) + 1e-10) / 5)
    macd_signal = (macd_signal + 1) / 2
    
    # 3. ADX
    delta = close_series.diff().values
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    plus_di = pd.Series(up).rolling(window=adx_period).sum().values / \
              (pd.Series(np.abs(delta)).rolling(window=adx_period).sum().values + 1e-10) * 100
    minus_di = pd.Series(down).rolling(window=adx_period).sum().values / \
               (pd.Series(np.abs(delta)).rolling(window=adx_period).sum().values + 1e-10) * 100
    adx_signal = np.clip(np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10), 0, 1)
    
    # 加權組合
    w_ema = {trend_formula.w_ema:.4f}
    w_macd = {trend_formula.w_macd:.4f}
    w_adx = {trend_formula.w_adx:.4f}
    
    trend = w_ema * ema_signal + w_macd * macd_signal + w_adx * adx_signal
    return np.clip(trend, 0, 1)


# ============================================================================
# 公式 3: 方向公式
# ============================================================================

def direction_formula(df):
    \"\"\"
    方向公式 - 預測下一步價格方向的確定性
    
    輸入: DataFrame with columns ['high', 'low', 'close']
    輸出: [0, 1] 的方向分數
      0   = 確定看跌
      0.5 = 中性
      1   = 確定看漲
    \"\"\"
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    close_series = pd.Series(close)
    
    # 參數
    rsi_period = {direction_formula.rsi_period}
    stoch_k = {direction_formula.stoch_k}
    stoch_d = {direction_formula.stoch_d}
    roc_period = {direction_formula.roc_period}
    
    # 1. RSI
    delta = close_series.diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_signal = rsi / 100
    
    # 2. Stochastic
    low_min = close_series.rolling(window=stoch_k).min().values
    high_max = close_series.rolling(window=stoch_k).max().values
    stoch_k_val = (close - low_min) / (high_max - low_min + 1e-10) * 100
    stoch_d_val = pd.Series(stoch_k_val).rolling(window=stoch_d).mean().values
    stoch_signal = stoch_d_val / 100
    
    # 3. ROC
    roc = (close - np.roll(close, roc_period)) / \
          (np.roll(close, roc_period) + 1e-10) * 100
    roc_signal = np.tanh(roc / 10)
    roc_signal = (roc_signal + 1) / 2
    
    # 加權組合
    w_rsi = {direction_formula.w_rsi:.4f}
    w_stoch = {direction_formula.w_stoch:.4f}
    w_roc = {direction_formula.w_roc:.4f}
    
    direction = w_rsi * rsi_signal + w_stoch * stoch_signal + w_roc * roc_signal
    return np.clip(direction, 0, 1)


# ============================================================================
# 使用範例
# ============================================================================

if __name__ == "__main__":
    # 假設 df 是包含 ['high', 'low', 'close'] 的 DataFrame
    
    vol = volatility_formula(df)
    trend = trend_formula(df)
    direction = direction_formula(df)
    
    # 構建特徵矩陣用於機器學習
    features = np.column_stack([
        df['close'].values,  # 價格
        vol,                  # 波動性
        trend,                # 趨勢
        direction             # 方向
    ])
    
    # features shape: (n_samples, 4)
    # 下一步: 用這些特徵訓練機器學習模型進行預測
"""
    
    with open('results/formula_code.py', 'w', encoding='utf-8') as f:
        f.write(formulas_code)
    
    print(f"✓ 公式代碼已保存: results/formula_code.py")
    
    print("\n" + "#" * 80)
    print("完成!")
    print("\n接下來: 用這 3 個特徵 + 價格特徵訓練機器學習模型")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
