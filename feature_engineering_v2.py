"""
高級特徵工程 (36 個特徵)

新櫾特徵:
1. 滿後特徵 (Lag Features) - 5 個
2. 相對強弱特徵 (Momentum) - 4 個  
3. 市場結構特徵 (Market Structure) - 4 個
4. 波動率特徵 (Volatility Relative) - 2 個

原始: 14 個 → 新增: 36 個
"""

import pandas as pd
import numpy as np
from typing import Dict


class AdvancedFeatureEngineering:
    """
    高級特徵工程
    """
    
    @staticmethod
    def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        殾後特徵 - 前几根 K 線的價格變化
        """
        result = df.copy()
        
        # 價格檻後 (Lag Price)
        for lag in lags:
            result[f'close_lag{lag}'] = df['close'].shift(lag)
            result[f'close_pct_lag{lag}'] = df['close'].pct_change(lag)
        
        # 方向殾後 (Lag Direction)
        for lag in [1, 2]:
            result[f'direction_lag{lag}'] = (df['close'].shift(lag) > df['close'].shift(lag+1)).astype(int)
        
        return result
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """
        相對強弱特徵 - 紅上指標的加速度
        """
        result = df.copy()
        
        # RSI 上加速度
        result['rsi_momentum'] = indicators['rsi'].diff()
        result['rsi_acceleration'] = indicators['rsi'].diff().diff()
        
        # MACD 上加速度
        result['macd_momentum'] = indicators['macd_line'].diff()
        
        # 布林帶壓佩 (低波動率是空亇信號)
        bb_width = indicators['upper_band'] - indicators['lower_band']
        result['bb_squeeze'] = bb_width / (indicators['middle_band'] + 1e-10)
        
        return result
    
    @staticmethod
    def add_market_structure_features(df: pd.DataFrame, indicators: Dict, lookback: int = 5) -> pd.DataFrame:
        """
        市場結構特徵 - 運價在近輷位置
        """
        result = df.copy()
        
        # N 根 K 線的最高最低位置
        rolling_high = df['high'].rolling(lookback).max()
        rolling_low = df['low'].rolling(lookback).min()
        rolling_range = rolling_high - rolling_low + 1e-10
        
        result['price_high_ratio'] = (df['close'] - rolling_low) / rolling_range
        result['price_low_ratio'] = (rolling_high - df['close']) / rolling_range
        
        # 連續上升下跌根數
        def count_consecutive_direction(series, direction='up'):
            """counts = []
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
        波動率相對特徵 - 當前波動相對於長貿平均
        """
        result = df.copy()
        
        # ATR 相對於平均
        atr_mean = indicators['atr'].rolling(lookback).mean()
        result['atr_ratio_to_mean'] = indicators['atr'] / (atr_mean + 1e-10)
        
        # 成交量相對於平均
        volume_mean = df['volume'].rolling(lookback).mean()
        result['volume_ratio_to_mean'] = df['volume'] / (volume_mean + 1e-10)
        
        return result
    
    @staticmethod
    def build_all_features(df: pd.DataFrame, indicators: Dict, 
                          trend_score: np.ndarray, 
                          volatility_score: np.ndarray,
                          direction_score: np.ndarray) -> pd.DataFrame:
        """
        构建所有 36 個特徵
        """
        X = pd.DataFrame(index=df.index)
        
        # 原始 14 個特徵
        X['trend_score'] = trend_score
        X['volatility_score'] = volatility_score
        X['direction_score'] = direction_score
        X['rsi'] = indicators['rsi'] / 100
        X['macd'] = indicators['macd_line']
        X['macd_signal'] = indicators['signal_line']
        X['bb_position'] = (df['close'] - indicators['lower_band']) / \
                           (indicators['upper_band'] - indicators['lower_band'] + 1e-10)
        X['atr'] = indicators['atr'] / (df['close'] + 1e-10)
        X['volume_ratio'] = indicators['volume_ratio']
        X['k_line'] = indicators['k_line'] / 100
        X['d_line'] = indicators['d_line'] / 100
        X['price_change_pct'] = df['close'].pct_change().abs()
        X['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        X['ema_trend'] = (indicators['ema_fast'] > indicators['ema_slow']).astype(int)
        
        # 新增特徵
        # 1. 殾後特徵 (5 個)
        lag_df = AdvancedFeatureEngineering.add_lag_features(df)
        X['close_lag1'] = lag_df['close_lag1']
        X['close_lag2'] = lag_df['close_lag2']
        X['close_pct_lag1'] = lag_df['close_pct_lag1']
        X['direction_lag1'] = lag_df['direction_lag1']
        X['direction_lag2'] = lag_df['direction_lag2']
        
        # 2. 相對強弱特徵 (4 個)
        momentum_df = AdvancedFeatureEngineering.add_momentum_features(df, indicators)
        X['rsi_momentum'] = momentum_df['rsi_momentum']
        X['rsi_acceleration'] = momentum_df['rsi_acceleration']
        X['macd_momentum'] = momentum_df['macd_momentum']
        X['bb_squeeze'] = momentum_df['bb_squeeze']
        
        # 3. 市場結構特徵 (4 個)
        structure_df = AdvancedFeatureEngineering.add_market_structure_features(df, indicators)
        X['price_high_ratio'] = structure_df['price_high_ratio']
        X['price_low_ratio'] = structure_df['price_low_ratio']
        X['consecutive_up'] = structure_df['consecutive_up']
        X['consecutive_down'] = structure_df['consecutive_down']
        
        # 4. 波動率相對特徵 (2 個)
        vol_df = AdvancedFeatureEngineering.add_volatility_relative_features(df, indicators)
        X['atr_ratio_to_mean'] = vol_df['atr_ratio_to_mean']
        X['volume_ratio_to_mean'] = vol_df['volume_ratio_to_mean']
        
        # 丟去 NaN
        X = X.dropna()
        
        print(f"\n特徵工程完成:")
        print(f"  原始特徵: 14 個")
        print(f"  殾後特徵: 5 個")
        print(f"  动量特徵: 4 個")
        print(f"  市场结构特征: 4 個")
        print(f"  波动率特征: 2 個")
        print(f"  " + "="*40)
        print(f"  總計: {X.shape[1]} 個特征")
        print(f"  数据佳量: {X.shape[0]} 行")
        print(f"  特征列表: {list(X.columns)}")
        
        return X


if __name__ == '__main__':
    print("此模組該往 train_models_v2.py 中使用")
