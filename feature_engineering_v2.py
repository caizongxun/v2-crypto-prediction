"""
高級特徵工程 (36 個特徵)

新增特徵:
1. 滯後特徵 (Lag Features) - 5 個
2. 相對強弱特徵 (Momentum) - 4 個  
3. 市場結構特徵 (Market Structure) - 4 個
4. 波動率特徵 (Volatility Relative) - 2 個

原始: 14 個 新增: 22 個 = 36 個
"""

import pandas as pd
import numpy as np
from typing import Dict


class AdvancedFeatureEngineering:
    """
    高級特徵工程
    """
    
    @staticmethod
    def add_lag_features(df: pd.DataFrame, lags: list = [1, 2]) -> pd.DataFrame:
        """
        滯後特徵 - 前幾根 K 線的價格變化
        """
        result = df.copy()
        
        # 價格滯後 (Lag Price)
        result['close_lag1'] = df['close'].shift(1)
        result['close_lag2'] = df['close'].shift(2)
        result['close_pct_lag1'] = df['close'].pct_change(1)
        
        # 方向滯後 (Lag Direction)
        result['direction_lag1'] = (df['close'].shift(1) > df['close'].shift(2)).astype(int)
        result['direction_lag2'] = (df['close'].shift(2) > df['close'].shift(3)).astype(int)
        
        return result
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """
        相對強弱特徵 - 紅上指標的加速度
        """
        result = df.copy()
        
        # RSI 上加速度
        rsi = indicators.get('rsi', pd.Series(np.zeros(len(df))))
        result['rsi_momentum'] = rsi.diff().fillna(0)
        result['rsi_acceleration'] = rsi.diff().diff().fillna(0)
        
        # MACD 上加速度
        macd_line = indicators.get('macd_line', pd.Series(np.zeros(len(df))))
        result['macd_momentum'] = macd_line.diff().fillna(0)
        
        # 布林帶壓縮 (低波動率是空亂信號)
        bb_upper = indicators.get('bb_upper', pd.Series(np.zeros(len(df))))
        bb_lower = indicators.get('bb_lower', pd.Series(np.zeros(len(df))))
        bb_middle = indicators.get('bb_middle', pd.Series(np.zeros(len(df))) + 1)
        
        bb_width = bb_upper - bb_lower
        result['bb_squeeze'] = (bb_width / (bb_middle + 1e-10)).fillna(0)
        
        return result
    
    @staticmethod
    def add_market_structure_features(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """
        市場結構特徵 - 價格在近期位置
        """
        result = df.copy()
        
        # N 根 K 線的最高最低位置
        rolling_high = df['high'].rolling(lookback).max()
        rolling_low = df['low'].rolling(lookback).min()
        rolling_range = rolling_high - rolling_low + 1e-10
        
        result['price_high_ratio'] = ((df['close'] - rolling_low) / rolling_range).fillna(0)
        result['price_low_ratio'] = ((rolling_high - df['close']) / rolling_range).fillna(0)
        
        # 連續上升下跌根數
        def count_consecutive_direction(series, direction='up'):
            counts = []
            count = 0
            for val in series:
                if (direction == 'up' and val > 0) or (direction == 'down' and val < 0):
                    count += 1
                else:
                    count = 0
                counts.append(count)
            return pd.Series(counts, index=series.index)
        
        price_change = df['close'].diff()
        result['consecutive_up'] = count_consecutive_direction(price_change, 'up')
        result['consecutive_down'] = count_consecutive_direction(price_change, 'down')
        
        return result
    
    @staticmethod
    def add_volatility_relative_features(df: pd.DataFrame, indicators: Dict, lookback: int = 20) -> pd.DataFrame:
        """
        波動率相對特徵 - 當前波動相對於長期平均
        """
        result = df.copy()
        
        # ATR 相對於平均
        atr = indicators.get('atr', pd.Series(np.zeros(len(df))))
        atr_mean = atr.rolling(lookback).mean()
        result['atr_ratio_to_mean'] = (atr / (atr_mean + 1e-10)).fillna(1)
        
        # 成交量相對於平均
        volume_mean = df['volume'].rolling(lookback).mean()
        result['volume_ratio_to_mean'] = (df['volume'] / (volume_mean + 1e-10)).fillna(1)
        
        return result
    
    @staticmethod
    def build_all_features(df: pd.DataFrame, indicators: Dict, 
                          trend_score: np.ndarray, 
                          volatility_score: np.ndarray,
                          direction_score: np.ndarray) -> pd.DataFrame:
        """
        構建所有 36 個特徵
        """
        X = pd.DataFrame(index=df.index)
        
        # 原始 14 個特徵
        print("\n構建特徵...")
        print("  第 1 部分: 基礎 14 個特徵")
        
        X['trend_score'] = trend_score
        X['volatility_score'] = volatility_score
        X['direction_score'] = direction_score
        X['rsi'] = indicators.get('rsi', 0) / 100
        X['macd'] = indicators.get('macd_line', 0)
        X['macd_signal'] = indicators.get('signal_line', 0)
        X['atr'] = indicators.get('atr', 0) / (df['close'] + 1e-10)
        X['volume_ratio'] = indicators.get('volume_sma', 0) / (df['volume'] + 1e-10)
        X['k_line'] = indicators.get('stochastic_k', 0) / 100
        X['d_line'] = indicators.get('stochastic_d', 0) / 100
        X['price_change_pct'] = df['close'].pct_change().abs().fillna(0)
        X['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        X['ema_trend'] = (indicators.get('ema_12', 0) > indicators.get('ema_26', 0)).astype(int)
        
        # BB 位置
        bb_upper = indicators.get('bb_upper', df['close'])
        bb_lower = indicators.get('bb_lower', df['close'])
        bb_range = bb_upper - bb_lower + 1e-10
        X['bb_position'] = ((df['close'] - bb_lower) / bb_range).fillna(0.5)
        
        # 新增 22 個特徵
        print("  第 2 部分: 滯後特徵 (5 個)")
        lag_df = AdvancedFeatureEngineering.add_lag_features(df)
        X['close_lag1'] = lag_df['close_lag1'].fillna(0)
        X['close_lag2'] = lag_df['close_lag2'].fillna(0)
        X['close_pct_lag1'] = lag_df['close_pct_lag1'].fillna(0)
        X['direction_lag1'] = lag_df['direction_lag1'].fillna(0)
        X['direction_lag2'] = lag_df['direction_lag2'].fillna(0)
        
        print("  第 3 部分: 動量特徵 (4 個)")
        momentum_df = AdvancedFeatureEngineering.add_momentum_features(df, indicators)
        X['rsi_momentum'] = momentum_df['rsi_momentum'].fillna(0)
        X['rsi_acceleration'] = momentum_df['rsi_acceleration'].fillna(0)
        X['macd_momentum'] = momentum_df['macd_momentum'].fillna(0)
        X['bb_squeeze'] = momentum_df['bb_squeeze'].fillna(0)
        
        print("  第 4 部分: 市場結構特徵 (4 個)")
        structure_df = AdvancedFeatureEngineering.add_market_structure_features(df)
        X['price_high_ratio'] = structure_df['price_high_ratio'].fillna(0)
        X['price_low_ratio'] = structure_df['price_low_ratio'].fillna(0)
        X['consecutive_up'] = structure_df['consecutive_up'].fillna(0)
        X['consecutive_down'] = structure_df['consecutive_down'].fillna(0)
        
        print("  第 5 部分: 波動率特徵 (2 個)")
        vol_df = AdvancedFeatureEngineering.add_volatility_relative_features(df, indicators)
        X['atr_ratio_to_mean'] = vol_df['atr_ratio_to_mean'].fillna(1)
        X['volume_ratio_to_mean'] = vol_df['volume_ratio_to_mean'].fillna(1)
        
        # 最終檢查
        X = X.fillna(0)
        
        print(f"\n特徵工程完成:")
        print(f"  基礎特徵: 14 個")
        print(f"  滯後特徵: 5 個")
        print(f"  動量特徵: 4 個")
        print(f"  市場結構: 4 個")
        print(f"  波動率: 2 個")
        print(f"  ========================")
        print(f"  總計: {X.shape[1]} 個特徵")
        print(f"  數據行數: {X.shape[0]} 行")
        print(f"  缺失值: {X.isnull().sum().sum()} 個")
        
        # 驗證特徵數量
        if X.shape[1] != 36:
            print(f"\n警告: 特徵數量不符! 預期 36 個, 實際 {X.shape[1]} 個")
            print(f"  特徵列表: {list(X.columns)}")
        
        return X


if __name__ == '__main__':
    print("此模組應當往 train_models_v2.py 中使用")
