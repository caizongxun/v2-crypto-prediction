import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class IndicatorCalculator:
    """計算交易視圖指標的輔助類"""
    
    @staticmethod
    def identify_swing_highs_lows(df: pd.DataFrame, period: int = 3) -> Tuple[List[int], List[int]]:
        """
        識別搖擺高點和低點
        
        Args:
            df: 包含 OHLCV 數據的 DataFrame
            period: 用於識別搖擺點的周期
            
        Returns:
            (swing_highs_indices, swing_lows_indices)
        """
        swing_highs = []
        swing_lows = []
        
        for i in range(period, len(df) - period):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            # 檢查搖擺高點
            is_high = True
            for j in range(1, period + 1):
                if high <= df['high'].iloc[i - j] or high <= df['high'].iloc[i + j]:
                    is_high = False
                    break
            
            if is_high:
                swing_highs.append(i)
            
            # 檢查搖擺低點
            is_low = True
            for j in range(1, period + 1):
                if low >= df['low'].iloc[i - j] or low >= df['low'].iloc[i + j]:
                    is_low = False
                    break
            
            if is_low:
                swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    @staticmethod
    def calculate_fibonacci_levels(high_idx: int, low_idx: int, high_price: float, 
                                   low_price: float, is_bullish: bool = True) -> Dict[str, float]:
        """
        計算斐波那契回調級別
        
        Args:
            high_idx: 高點索引
            low_idx: 低點索引
            high_price: 高點價格
            low_price: 低點價格
            is_bullish: 是否為看漲結構
            
        Returns:
            斐波那契級別字典 {level: price}
        """
        fib_levels = {
            '0.0': 0.0,
            '0.236': 0.236,
            '0.382': 0.382,
            '0.5': 0.5,
            '0.618': 0.618,
            '0.786': 0.786,
            '1.0': 1.0
        }
        
        range_price = abs(high_price - low_price)
        
        fib_values = {}
        for level_str, level_val in fib_levels.items():
            if is_bullish:
                # 看漲：從高點向下
                price = high_price - (range_price * level_val)
            else:
                # 看跌：從低點向上
                price = low_price + (range_price * level_val)
            
            fib_values[level_str] = price
        
        return fib_values
    
    @staticmethod
    def identify_order_blocks(df: pd.DataFrame, periods: int = 5, 
                             threshold: float = 0.0) -> List[Dict]:
        """
        識別訂單塊
        
        Args:
            df: 包含 OHLCV 數據的 DataFrame
            periods: 識別訂單塊所需的連續蠟燭數
            threshold: 最小百分比變動閾值
            
        Returns:
            訂單塊列表，每個包含 {'type', 'index', 'high', 'low', 'avg'}
        """
        order_blocks = []
        ob_period = periods + 1
        
        for i in range(ob_period, len(df)):
            # 計算絕對百分比變動
            abs_move = (abs(df['close'].iloc[i] - df['close'].iloc[i - periods]) / 
                       df['close'].iloc[i - ob_period]) * 100
            rel_move = abs_move >= threshold
            
            if not rel_move:
                continue
            
            # 看漲訂單塊：最後一根紅蠟燭 + 後續綠蠟燭序列
            bullish_ob = df['close'].iloc[i - ob_period] < df['open'].iloc[i - ob_period]
            up_candles = 0
            
            for j in range(1, periods + 1):
                if df['close'].iloc[i - j] > df['open'].iloc[i - j]:
                    up_candles += 1
            
            if bullish_ob and up_candles == periods:
                ob_high = df['open'].iloc[i - ob_period]
                ob_low = df['low'].iloc[i - ob_period]
                ob_avg = (ob_high + ob_low) / 2
                
                order_blocks.append({
                    'type': 'bullish',
                    'index': i - ob_period,
                    'high': ob_high,
                    'low': ob_low,
                    'avg': ob_avg
                })
            
            # 看跌訂單塊：最後一根綠蠟燭 + 後續紅蠟燭序列
            bearish_ob = df['close'].iloc[i - ob_period] > df['open'].iloc[i - ob_period]
            down_candles = 0
            
            for j in range(1, periods + 1):
                if df['close'].iloc[i - j] < df['open'].iloc[i - j]:
                    down_candles += 1
            
            if bearish_ob and down_candles == periods:
                ob_high = df['high'].iloc[i - ob_period]
                ob_low = df['open'].iloc[i - ob_period]
                ob_avg = (ob_high + ob_low) / 2
                
                order_blocks.append({
                    'type': 'bearish',
                    'index': i - ob_period,
                    'high': ob_high,
                    'low': ob_low,
                    'avg': ob_avg
                })
        
        return order_blocks
    
    @staticmethod
    def identify_market_structure(df: pd.DataFrame) -> List[Dict]:
        """
        識別市場結構（低位更低的低位 LL，高位更高的高位 HH等）
        
        Args:
            df: 包含 OHLCV 數據的 DataFrame
            
        Returns:
            市場結構列表
        """
        structures = []
        
        # 簡化版：識別簡單的結構
        for i in range(2, len(df) - 1):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            prev_high = df['high'].iloc[i - 1]
            prev_low = df['low'].iloc[i - 1]
            prev_prev_high = df['high'].iloc[i - 2]
            prev_prev_low = df['low'].iloc[i - 2]
            
            # 高位更高 (Higher High)
            if current_high > prev_high > prev_prev_high:
                structures.append({
                    'type': 'HH',
                    'index': i,
                    'price': current_high
                })
            
            # 高位更低 (Lower High)
            elif current_high < prev_high < prev_prev_high:
                structures.append({
                    'type': 'LH',
                    'index': i,
                    'price': current_high
                })
            
            # 低位更高 (Higher Low)
            if current_low > prev_low > prev_prev_low:
                structures.append({
                    'type': 'HL',
                    'index': i,
                    'price': current_low
                })
            
            # 低位更低 (Lower Low)
            elif current_low < prev_low < prev_prev_low:
                structures.append({
                    'type': 'LL',
                    'index': i,
                    'price': current_low
                })
        
        return structures
