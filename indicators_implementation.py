#!/usr/bin/env python3
"""
三個指標的完整實現 - 包含所有數學公式

1. 趨勢指標 (Trend Indicator)
2. 方向指標 (Direction Indicator)
3. 波動性指標 (Volatility Indicator)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import json
import os


class TrendIndicator:
    """
    趨勢指標實現
    
    公式:
    -----
    1. fast_ema = EMA(close, 23)
    2. slow_ema = EMA(close, 90)
    3. tr = max(high - low, |high - close[t-1]|, |low - close[t-1]|)
    4. atr = EMA(tr, 10)
    5. ema_ratio = (fast_ema - slow_ema) / slow_ema * 100
    6. atr_ratio = (atr / close) * 100
    7. trend_score = tanh(ema_ratio / 2) * (1 - exp(-atr_ratio / 0.5))
    8. trend_value = (trend_score + 1) / 2  # 正規化 [0, 1]
    
    解釋:
    ----
    - trend_value > 0.6: 強上漲趨勢 (買入信號)
    - trend_value > 0.5: 上漲趨勢
    - 0.4 <= trend_value <= 0.5: 中立
    - trend_value < 0.4: 下跌趨勢
    - trend_value < 0.3: 強下跌趨勢 (賣出信號)
    """
    
    FAST_EMA_PERIOD = 23
    SLOW_EMA_PERIOD = 90
    ATR_PERIOD = 10
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.Series:
        """
        計算趨勢指標
        
        Args:
            df: DataFrame with 'close', 'high', 'low' columns
        
        Returns:
            pd.Series: 趨勢指標值 [0, 1]
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Step 1: 計算 EMA
        # EMA = 指數移動平均
        # 公式: EMA(t) = close(t) * 2/(n+1) + EMA(t-1) * (1 - 2/(n+1))
        fast_ema = close.ewm(span=TrendIndicator.FAST_EMA_PERIOD, adjust=False).mean()
        slow_ema = close.ewm(span=TrendIndicator.SLOW_EMA_PERIOD, adjust=False).mean()
        
        # Step 2: 計算真實波幅 (True Range)
        # TR = max(H-L, |H-C(t-1)|, |L-C(t-1)|)
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Step 3: 計算 ATR (平均真實波幅)
        # ATR = EMA(TR, period)
        atr = tr.ewm(span=TrendIndicator.ATR_PERIOD, adjust=False).mean()
        
        # Step 4: 計算 EMA 比率
        # EMA_ratio = (fast_ema - slow_ema) / slow_ema * 100
        # 說明: 快速 EMA 與慢速 EMA 的百分比差異
        ema_ratio = (fast_ema - slow_ema) / slow_ema * 100
        
        # Step 5: 計算 ATR 比率
        # ATR_ratio = (ATR / close) * 100
        # 說明: 波動性相對於收盤價的百分比
        atr_ratio = (atr / close) * 100
        
        # Step 6: 計算趨勢分數
        # trend_score = tanh(ema_ratio / 2) * (1 - exp(-atr_ratio / 0.5))
        # 說明:
        #   - tanh() 函數將 EMA 比率正規化至 [-1, 1]
        #   - (1 - exp(-x)) 將 ATR 比率作為權重
        #   - 兩者相乘得到最終的趨勢分數
        trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-atr_ratio / 0.5))
        
        # Step 7: 正規化趨勢值
        # trend_value = (trend_score + 1) / 2
        # 說明: 將 [-1, 1] 範圍轉換為 [0, 1] 範圍
        trend_value = (trend_score + 1) / 2
        
        return trend_value.clip(0, 1)
    
    @staticmethod
    def get_signal(trend_value: float) -> str:
        """
        根據趨勢值生成信號
        """
        if trend_value > 0.6:
            return "強上漲"
        elif trend_value > 0.5:
            return "上漲"
        elif trend_value < 0.3:
            return "強下跌"
        elif trend_value < 0.4:
            return "下跌"
        else:
            return "中立"


class DirectionIndicator:
    """
    方向指標實現
    
    公式:
    -----
    1. delta = close(t) - close(t-1)
    2. gain = max(delta, 0)
    3. loss = max(-delta, 0)
    4. avg_gain = EMA(gain, 10)
    5. avg_loss = EMA(loss, 10)
    6. rs = avg_gain / avg_loss
    7. rsi = 100 - (100 / (1 + rs))
    8. roc = ((close - close[t-8]) / close[t-8]) * 100
    9. rsi_signal = (rsi - 50) / 50
    10. roc_signal = tanh(roc / 5)
    11. direction_score = (rsi_signal + roc_signal) / 2
    12. direction_value = (direction_score + 1) / 2  # 正規化 [0, 1]
    
    解釋:
    ----
    - direction_value > 0.6: 強上漲方向 (買入)
    - direction_value > 0.5: 上漲方向
    - 0.4 <= direction_value <= 0.5: 中立
    - direction_value < 0.4: 下跌方向
    - direction_value < 0.3: 強下跌方向 (賣出)
    """
    
    RSI_PERIOD = 10
    ROC_PERIOD = 8
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.Series:
        """
        計算方向指標
        
        Args:
            df: DataFrame with 'close' column
        
        Returns:
            pd.Series: 方向指標值 [0, 1]
        """
        close = df['close']
        
        # Step 1: 計算每日價格變化
        # delta = close(t) - close(t-1)
        delta = close.diff()
        
        # Step 2: 分離上漲和下跌
        # gain = max(delta, 0)  # 只保留正數
        # loss = max(-delta, 0)  # 只保留負數的絕對值
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Step 3: 計算平均收益和平均虧損
        # avg_gain = EMA(gain, period)
        # avg_loss = EMA(loss, period)
        avg_gain = pd.Series(gain).ewm(span=DirectionIndicator.RSI_PERIOD, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(span=DirectionIndicator.RSI_PERIOD, adjust=False).mean()
        
        # Step 4: 計算相對強弱指數 (RSI)
        # rs = avg_gain / avg_loss
        # rsi = 100 - (100 / (1 + rs))
        # 說明: RSI 測量上漲與下跌的相對強度
        #       值域 [0, 100]
        #       > 70: 超買
        #       < 30: 超賣
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Step 5: 計算價格變動率 (ROC)
        # roc = ((close(t) - close(t-n)) / close(t-n)) * 100
        # 說明: 測量價格在 n 個週期內的百分比變化
        roc = ((close - close.shift(DirectionIndicator.ROC_PERIOD)) / close.shift(DirectionIndicator.ROC_PERIOD)) * 100
        
        # Step 6: 正規化 RSI 信號
        # rsi_signal = (rsi - 50) / 50
        # 說明: 將 [0, 100] 轉換為 [-1, 1]
        #       > 0: 看漲
        #       < 0: 看跌
        rsi_signal = (rsi - 50) / 50
        
        # Step 7: 正規化 ROC 信號
        # roc_signal = tanh(roc / 5)
        # 說明: 將 ROC 正規化至 [-1, 1]
        roc_signal = np.tanh(roc / 5)
        
        # Step 8: 計算方向分數
        # direction_score = (rsi_signal + roc_signal) / 2
        # 說明: 結合 RSI 和 ROC 的信號
        direction_score = (rsi_signal + roc_signal) / 2
        
        # Step 9: 正規化方向值
        # direction_value = (direction_score + 1) / 2
        # 說明: 將 [-1, 1] 轉換為 [0, 1]
        direction_value = (direction_score + 1) / 2
        
        return direction_value.clip(0, 1)
    
    @staticmethod
    def get_signal(direction_value: float) -> str:
        """
        根據方向值生成信號
        """
        if direction_value > 0.6:
            return "強上漲"
        elif direction_value > 0.5:
            return "上漲"
        elif direction_value < 0.3:
            return "強下跌"
        elif direction_value < 0.4:
            return "下跌"
        else:
            return "中立"


class VolatilityIndicator:
    """
    波動性指標實現
    
    公式:
    -----
    1. sma = SMA(close, 39)
    2. std = STDEV(close, 39)
    3. upper = sma + (2.6 * std)
    4. lower = sma - (2.6 * std)
    5. volatility_ratio = (std / sma) * 100
    6. volatility_score = sqrt(volatility_ratio / 2)
    7. volatility_value = min(volatility_score, 1.0)  # 正規化 [0, 1]
    
    解釋:
    ----
    - volatility_value > 0.6: 高波動性 (適合操作)
    - volatility_value > 0.4: 中低波動性
    - volatility_value < 0.3: 低波動性 (盤整狀態)
    """
    
    SMA_PERIOD = 39
    BB_STD_MULTIPLIER = 2.6
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.Series:
        """
        計算波動性指標
        
        Args:
            df: DataFrame with 'close' column
        
        Returns:
            pd.Series: 波動性指標值 [0, 1]
        """
        close = df['close']
        
        # Step 1: 計算簡單移動平均
        # sma = SMA(close, period)
        # 說明: 計算過去 n 個收盤價的平均值
        sma = close.rolling(window=VolatilityIndicator.SMA_PERIOD).mean()
        
        # Step 2: 計算標準差
        # std = STDEV(close, period)
        # 說明: 衡量價格在平均值周圍的分散程度
        std = close.rolling(window=VolatilityIndicator.SMA_PERIOD).std()
        
        # Step 3: 計算布林通道 (Bollinger Bands)
        # upper = sma + (k * std)
        # lower = sma - (k * std)
        # 說明: k 通常為 2，表示 95% 的數據應在此範圍內
        #       我們使用 2.6 以獲得更寬的波道
        upper = sma + (VolatilityIndicator.BB_STD_MULTIPLIER * std)
        lower = sma - (VolatilityIndicator.BB_STD_MULTIPLIER * std)
        
        # Step 4: 計算波動性比率
        # volatility_ratio = (std / sma) * 100
        # 說明: 衡量相對波動性 (%).
        #       > 2%: 高波動性
        #       < 1%: 低波動性
        volatility_ratio = (std / sma) * 100
        
        # Step 5: 計算波動性分數
        # volatility_score = sqrt(volatility_ratio / 2)
        # 說明: 使用平方根函數來平滑數據
        volatility_score = np.sqrt(volatility_ratio / 2)
        
        # Step 6: 正規化波動性值
        # volatility_value = min(volatility_score, 1.0)
        # 說明: 將值限制在 [0, 1] 範圍內
        volatility_value = volatility_score.clip(0, 1)
        
        return volatility_value
    
    @staticmethod
    def get_signal(volatility_value: float) -> str:
        """
        根據波動性值生成信號
        """
        if volatility_value > 0.6:
            return "高波動性 (適合交易)"
        elif volatility_value > 0.4:
            return "中低波動性"
        else:
            return "低波動性 (盤整狀態)"


def main():
    """
    主程序 - 演示三個指標的使用
    """
    print("\n" + "="*80)
    print("三個指標的完整實現 - 包含所有數學公式")
    print("="*80)
    
    # 加載數據
    print("\n[1] 加載數據...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        
        # 過濾日期
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        print(f"✓ 數據加載成功")
        print(f"  時間範圍: {df.index[0]} ~ {df.index[-1]}")
        print(f"  K線數量: {len(df)}")
    except Exception as e:
        print(f"✗ 數據加載失敗: {e}")
        return
    
    # 計算三個指標
    print("\n[2] 計算指標...")
    
    print("\n  計算趨勢指標 (Trend Indicator)...")
    trend = TrendIndicator.calculate(df)
    print(f"  ✓ 計算完成 - 值域: [{trend.min():.4f}, {trend.max():.4f}]")
    
    print("\n  計算方向指標 (Direction Indicator)...")
    direction = DirectionIndicator.calculate(df)
    direction = direction.dropna()
    print(f"  ✓ 計算完成 - 值域: [{direction.min():.4f}, {direction.max():.4f}]")
    
    print("\n  計算波動性指標 (Volatility Indicator)...")
    volatility = VolatilityIndicator.calculate(df)
    print(f"  ✓ 計算完成 - 值域: [{volatility.min():.4f}, {volatility.max():.4f}]")
    
    # 顯示最新的指標值
    print("\n" + "="*80)
    print("最新指標值 (2024-12-31)")
    print("="*80)
    
    latest_date = df.index[-1]
    latest_trend = trend.iloc[-1]
    latest_direction = direction.iloc[-1]
    latest_volatility = volatility.iloc[-1]
    
    print(f"\n時間: {latest_date}")
    print(f"\n趨勢指標: {latest_trend:.4f} ({TrendIndicator.get_signal(latest_trend)})")
    print(f"方向指標: {latest_direction:.4f} ({DirectionIndicator.get_signal(latest_direction)})")
    print(f"波動性指標: {latest_volatility:.4f} ({VolatilityIndicator.get_signal(latest_volatility)})")
    
    # 保存完整公式說明
    print("\n[3] 保存公式說明...")
    os.makedirs("results", exist_ok=True)
    
    formulas = {
        "trend": {
            "description": TrendIndicator.__doc__,
            "parameters": {
                "fast_ema_period": TrendIndicator.FAST_EMA_PERIOD,
                "slow_ema_period": TrendIndicator.SLOW_EMA_PERIOD,
                "atr_period": TrendIndicator.ATR_PERIOD
            }
        },
        "direction": {
            "description": DirectionIndicator.__doc__,
            "parameters": {
                "rsi_period": DirectionIndicator.RSI_PERIOD,
                "roc_period": DirectionIndicator.ROC_PERIOD
            }
        },
        "volatility": {
            "description": VolatilityIndicator.__doc__,
            "parameters": {
                "sma_period": VolatilityIndicator.SMA_PERIOD,
                "bb_std_multiplier": VolatilityIndicator.BB_STD_MULTIPLIER
            }
        }
    }
    
    with open("results/indicators_formulas.json", "w", encoding="utf-8") as f:
        json.dump(formulas, f, indent=2, ensure_ascii=False, default=str)
    
    print("✓ 公式已保存至: results/indicators_formulas.json")
    
    # 保存計算結果
    print("\n[4] 保存計算結果...")
    results_df = pd.DataFrame({
        'trend': trend,
        'direction': direction,
        'volatility': volatility
    })
    
    results_df.to_csv("results/indicators_values.csv")
    print("✓ 結果已保存至: results/indicators_values.csv")
    
    print("\n" + "="*80)
    print("完成")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
