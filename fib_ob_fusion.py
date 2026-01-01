import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class FibonacciBollingerBands:
    """
    Fibonacci Bollinger Bands: Modified Bollinger Bands using Fibonacci ratios
    
    Variables meaning:
    - length: Period for basis calculation (default: 200)
    - src: Source price (default: HLC3 = (high + low + close) / 3)
    - mult: Standard deviation multiplier (default: 3.0)
    - basis: Volume Weighted Moving Average (VWMA) of source
    - dev: Standard deviation of source * multiplier
    - Fibonacci levels: 0.236, 0.382, 0.5, 0.618, 0.764, 1.0
    """
    
    def __init__(self, length=200, mult=3.0):
        self.length = length
        self.mult = mult
        # Fibonacci ratios for band levels
        self.fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.764, 1.0]
    
    def _vwma(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """
        Calculate Volume Weighted Moving Average
        """
        if 'volume' not in df.columns:
            return df['close'].rolling(window=period).mean().values
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume = df['volume']
        
        vwma = pd.Series(index=df.index, dtype=float)
        for i in range(period - 1, len(df)):
            start = max(0, i - period + 1)
            end = i + 1
            vwma.iloc[i] = np.average(
                typical_price.iloc[start:end],
                weights=volume.iloc[start:end]
            )
        
        return vwma.values
    
    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Fibonacci Bollinger Bands
        
        Returns:
            Dictionary with basis and upper/lower bands at Fibonacci levels
        """
        # Calculate basis (VWMA)
        basis = self._vwma(df, self.length)
        
        # Calculate standard deviation
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        dev = typical_price.rolling(window=self.length).std().values * self.mult
        
        # Initialize result dictionary
        result = {
            'basis': basis,
            'upper_bands': {},
            'lower_bands': {},
        }
        
        # Calculate Fibonacci bands
        for ratio in self.fib_ratios:
            result['upper_bands'][ratio] = basis + (ratio * dev)
            result['lower_bands'][ratio] = basis - (ratio * dev)
        
        return result


class OrderBlockDetector:
    """
    Order Block Detection: Identify institutional order blocks
    
    Variables meaning:
    - periods: Required consecutive candles to identify OB (default: 5)
    - threshold: Minimum percent move required (default: 0.0%)
    - usewicks: Use full High/Low range or Open/Low for Bullish, Open/High for Bearish
    
    Logic:
    - Bullish OB: Last down candle before sequence of up candles (price range: Open to Low)
    - Bearish OB: Last up candle before sequence of down candles (price range: Open to High)
    """
    
    def __init__(self, periods=5, threshold=0.0, usewicks=False):
        self.periods = periods
        self.threshold = threshold
        self.usewicks = usewicks
        self.ob_period = periods + 1
    
    def _detect_bullish_ob(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Bullish Order Blocks
        Bullish OB: Last RED candle before sequence of GREEN candles
        """
        bullish_obs = []
        
        for i in range(self.ob_period, len(df)):
            # Check if candle at ob_period position is RED (close < open)
            ob_candle_idx = i - self.ob_period
            if df['close'].iloc[ob_candle_idx] >= df['open'].iloc[ob_candle_idx]:
                continue
            
            # Count consecutive GREEN candles after OB
            green_count = 0
            for j in range(i - self.periods, i):
                if df['close'].iloc[j] > df['open'].iloc[j]:
                    green_count += 1
            
            if green_count != self.periods:
                continue
            
            # Check percent move threshold
            percent_move = abs(df['close'].iloc[i - 1] - df['close'].iloc[ob_candle_idx]) / df['close'].iloc[ob_candle_idx] * 100
            if percent_move < self.threshold:
                continue
            
            # Calculate OB levels
            segment = df.iloc[ob_candle_idx:i]
            high = segment['high'].max()
            low = segment['low'].min()
            ob_high = df['open'].iloc[ob_candle_idx] if not self.usewicks else high
            ob_low = low
            ob_avg = (ob_high + ob_low) / 2
            
            bullish_obs.append({
                'type': 'bullish',
                'index': ob_candle_idx,
                'high': ob_high,
                'low': ob_low,
                'avg': ob_avg,
                'percent_move': percent_move,
                'price_at_ob': df['close'].iloc[ob_candle_idx]
            })
        
        return bullish_obs
    
    def _detect_bearish_ob(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Bearish Order Blocks
        Bearish OB: Last GREEN candle before sequence of RED candles
        """
        bearish_obs = []
        
        for i in range(self.ob_period, len(df)):
            # Check if candle at ob_period position is GREEN (close > open)
            ob_candle_idx = i - self.ob_period
            if df['close'].iloc[ob_candle_idx] <= df['open'].iloc[ob_candle_idx]:
                continue
            
            # Count consecutive RED candles after OB
            red_count = 0
            for j in range(i - self.periods, i):
                if df['close'].iloc[j] < df['open'].iloc[j]:
                    red_count += 1
            
            if red_count != self.periods:
                continue
            
            # Check percent move threshold
            percent_move = abs(df['close'].iloc[i - 1] - df['close'].iloc[ob_candle_idx]) / df['close'].iloc[ob_candle_idx] * 100
            if percent_move < self.threshold:
                continue
            
            # Calculate OB levels
            segment = df.iloc[ob_candle_idx:i]
            high = segment['high'].max()
            low = segment['low'].min()
            ob_high = high
            ob_low = df['open'].iloc[ob_candle_idx] if not self.usewicks else low
            ob_avg = (ob_high + ob_low) / 2
            
            bearish_obs.append({
                'type': 'bearish',
                'index': ob_candle_idx,
                'high': ob_high,
                'low': ob_low,
                'avg': ob_avg,
                'percent_move': percent_move,
                'price_at_ob': df['close'].iloc[ob_candle_idx]
            })
        
        return bearish_obs
    
    def detect(self, df: pd.DataFrame) -> Dict:
        """
        Detect all Order Blocks
        """
        bullish = self._detect_bullish_ob(df)
        bearish = self._detect_bearish_ob(df)
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'all': bullish + bearish
        }


class FibOBFusion:
    """
    Fusion of Fibonacci Bollinger Bands and Order Block indicators
    Creates combined features for machine learning
    """
    
    def __init__(self, fib_length=200, fib_mult=3.0, ob_periods=5, ob_threshold=0.0):
        self.fib = FibonacciBollingerBands(length=fib_length, mult=fib_mult)
        self.ob = OrderBlockDetector(periods=ob_periods, threshold=ob_threshold)
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform combined analysis
        """
        fib_result = self.fib.calculate(df)
        ob_result = self.ob.detect(df)
        
        return {
            'fib': fib_result,
            'ob': ob_result,
            'fusion': self._create_fusion_features(df, fib_result, ob_result)
        }
    
    def _create_fusion_features(self, df: pd.DataFrame, fib_result: Dict, ob_result: Dict) -> Dict:
        """
        Create features combining both indicators
        """
        fusion_data = {
            'price_to_basis': [],  # Distance from price to FBB basis
            'fib_level_zones': [],  # Which Fibonacci zones price is in
            'ob_near_fib': [],      # Whether OB is near Fibonacci levels
            'signal_strength': []   # Combined signal strength
        }
        
        basis = fib_result['basis']
        upper_1 = fib_result['upper_bands'][0.618]
        lower_1 = fib_result['lower_bands'][0.618]
        
        close_prices = df['close'].values
        
        for i in range(len(df)):
            if pd.isna(basis[i]):
                fusion_data['price_to_basis'].append(0)
                fusion_data['fib_level_zones'].append(0)
                fusion_data['ob_near_fib'].append(0)
                fusion_data['signal_strength'].append(0)
                continue
            
            price = close_prices[i]
            # Normalize distance to basis
            price_to_basis = (price - basis[i]) / (upper_1[i] - lower_1[i]) if not pd.isna(upper_1[i]) else 0
            fusion_data['price_to_basis'].append(price_to_basis)
            
            # Determine Fibonacci zone
            if price > upper_1[i]:
                zone = 1  # Above upper 0.618
            elif price < lower_1[i]:
                zone = -1  # Below lower 0.618
            else:
                zone = 0  # Within 0.618 bands
            fusion_data['fib_level_zones'].append(zone)
            
            # Check proximity to OBs
            ob_proximity = 0
            for ob in ob_result['all']:
                if ob['low'] <= price <= ob['high']:
                    ob_proximity += 1
            fusion_data['ob_near_fib'].append(min(ob_proximity, 3))  # Cap at 3
            
            # Combined signal: zone + OB proximity
            signal = abs(fusion_data['fib_level_zones'][-1]) + fusion_data['ob_near_fib'][-1]
            fusion_data['signal_strength'].append(signal)
        
        return fusion_data
