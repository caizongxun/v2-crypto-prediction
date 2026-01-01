import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class FibonacciBollingerBands:
    """Fibonacci Bollinger Bands Implementation
    
    Variables:
    - length: Period for VWMA basis calculation (200)
    - src: Source data (hlc3 = (high+low+close)/3)
    - mult: Multiplier for standard deviation (3.0)
    - basis: VWMA of source
    - dev: Standard deviation * multiplier
    - upper_bands: Dict with fibonacci ratios (0.236, 0.382, 0.5, 0.618, 0.764, 1.0)
    - lower_bands: Dict with fibonacci ratios (0.236, 0.382, 0.5, 0.618, 0.764, 1.0)
    """
    
    def __init__(self, length: int = 200, mult: float = 3.0):
        self.length = length
        self.mult = mult
        self.fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.764, 1.0]
        
    def calculate_vwma(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate Volume-Weighted Moving Average"""
        if 'volume' not in df.columns:
            return df['hlc3'].rolling(self.length).mean().values
        
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        volume = df['volume'].fillna(1)
        
        vwma = np.zeros(len(df))
        for i in range(len(df)):
            if i < self.length:
                window_prices = hlc3.iloc[:i+1].values
                window_volumes = volume.iloc[:i+1].values
                vwma[i] = np.average(window_prices, weights=window_volumes) if window_volumes.sum() > 0 else window_prices.mean()
            else:
                window_prices = hlc3.iloc[i-self.length+1:i+1].values
                window_volumes = volume.iloc[i-self.length+1:i+1].values
                vwma[i] = np.average(window_prices, weights=window_volumes) if window_volumes.sum() > 0 else window_prices.mean()
        
        return vwma
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze Fibonacci Bollinger Bands"""
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        basis = self.calculate_vwma(df)
        dev = np.std([hlc3.iloc[max(0, i-self.length+1):i+1].values for i in range(len(df))], axis=0) * self.mult
        
        upper_bands = {}
        lower_bands = {}
        
        for ratio in self.fib_ratios:
            upper_bands[ratio] = basis + (ratio * dev)
            lower_bands[ratio] = basis - (ratio * dev)
        
        return {
            'basis': basis,
            'dev': dev,
            'upper_bands': upper_bands,
            'lower_bands': lower_bands,
            'hlc3': hlc3.values,
        }


class OrderBlockDetector:
    """Order Block Detection Implementation
    
    Variables:
    - periods: Required consecutive candles to identify OB (5)
    - threshold: Minimum percent move to identify OB (0.0%)
    - use_wicks: Use full High/Low range instead of Open/Close (False)
    
    Logic:
    - Bullish OB: Last down candle (close < open) before 'periods' consecutive up candles (close > open)
    - Bearish OB: Last up candle (close > open) before 'periods' consecutive down candles (close < open)
    """
    
    def __init__(self, periods: int = 5, threshold: float = 0.0, use_wicks: bool = False):
        self.periods = periods
        self.threshold = threshold
        self.use_wicks = use_wicks
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Detect Order Blocks"""
        n = len(df)
        open_ = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        bullish_obs = []
        bearish_obs = []
        
        ob_period = self.periods + 1
        
        for i in range(ob_period, n):
            # Calculate absolute percent move
            if close[i - ob_period] != 0:
                abs_move = abs(close[i] - close[i - ob_period]) / close[i - ob_period] * 100
            else:
                abs_move = 0
            
            rel_move = abs_move >= self.threshold
            
            if not rel_move:
                continue
            
            # Bullish OB: Red candle (close < open) + subsequent green candles
            if close[i - ob_period] < open_[i - ob_period]:
                up_candles = sum(1 for j in range(i - self.periods, i) if close[j] > open_[j])
                
                if up_candles == self.periods:
                    ob_high = open_[i - ob_period] if not self.use_wicks else high[i - ob_period]
                    ob_low = low[i - ob_period]
                    ob_avg = (ob_high + ob_low) / 2
                    
                    bullish_obs.append({
                        'type': 'bullish',
                        'index': i - ob_period,
                        'high': ob_high,
                        'low': ob_low,
                        'avg': ob_avg,
                        'open': open_[i - ob_period],
                        'close': close[i - ob_period],
                        'percent_move': abs_move,
                    })
            
            # Bearish OB: Green candle (close > open) + subsequent red candles
            if close[i - ob_period] > open_[i - ob_period]:
                down_candles = sum(1 for j in range(i - self.periods, i) if close[j] < open_[j])
                
                if down_candles == self.periods:
                    ob_high = high[i - ob_period]
                    ob_low = open_[i - ob_period] if not self.use_wicks else low[i - ob_period]
                    ob_avg = (ob_high + ob_low) / 2
                    
                    bearish_obs.append({
                        'type': 'bearish',
                        'index': i - ob_period,
                        'high': ob_high,
                        'low': ob_low,
                        'avg': ob_avg,
                        'open': open_[i - ob_period],
                        'close': close[i - ob_period],
                        'percent_move': abs_move,
                    })
        
        return {
            'bullish': bullish_obs,
            'bearish': bearish_obs,
            'all': bullish_obs + bearish_obs,
        }


class FibOBFusion:
    """Fibonacci-OB Fusion: Learn relationship between Fibonacci levels and Order Blocks
    
    Hypothesis: Price tends to reverse or consolidate at Fibonacci levels when combined with OB.
    This class detects when:
    1. K-line is at Fibonacci band levels (0.236-0.764 or 1.0 limits)
    2. Order Blocks exist nearby
    3. Combined signal strength
    """
    
    def __init__(self, fib_length: int = 200, fib_mult: float = 3.0, ob_periods: int = 5):
        self.fib = FibonacciBollingerBands(length=fib_length, mult=fib_mult)
        self.ob_detector = OrderBlockDetector(periods=ob_periods, threshold=0.0)
        
        self.fib_length = fib_length
        self.fib_mult = fib_mult
        self.ob_periods = ob_periods
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Comprehensive fusion analysis"""
        fib_result = self.fib.analyze(df)
        ob_result = self.ob_detector.analyze(df)
        
        # Learn relationships
        relationships = self._learn_relationships(df, fib_result, ob_result)
        
        return {
            'fib': fib_result,
            'ob': ob_result,
            'relationships': relationships,
            'config': {
                'fib_length': self.fib_length,
                'fib_mult': self.fib_mult,
                'ob_periods': self.ob_periods,
            }
        }
    
    def _learn_relationships(self, df: pd.DataFrame, fib_result: Dict, ob_result: Dict) -> Dict:
        """Learn relationships between Fibonacci levels and Order Blocks"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        basis = fib_result['basis']
        upper_618 = fib_result['upper_bands'][0.618]
        lower_618 = fib_result['lower_bands'][0.618]
        upper_1 = fib_result['upper_bands'][1.0]
        lower_1 = fib_result['lower_bands'][1.0]
        
        signals = []
        
        # For each candlestick, detect interaction with fibonacci levels
        for i in range(len(df)):
            candle_high = high[i]
            candle_low = low[i]
            candle_close = close[i]
            
            # Check if candle touches fibonacci levels
            fib_level_interaction = self._check_fib_interaction(
                candle_high, candle_low, candle_close,
                basis[i], upper_618[i], lower_618[i], upper_1[i], lower_1[i]
            )
            
            if not fib_level_interaction:
                continue
            
            # Check if nearby OB exists
            nearby_obs = self._find_nearby_obs(i, ob_result, lookback=50)
            
            if nearby_obs:
                signal_strength = self._calculate_signal_strength(
                    i, fib_level_interaction, nearby_obs, candle_close, basis[i]
                )
                
                signals.append({
                    'index': i,
                    'close': candle_close,
                    'fib_interaction': fib_level_interaction,
                    'nearby_obs': nearby_obs,
                    'signal_strength': signal_strength,
                })
        
        return {
            'signals': signals,
            'signal_count': len(signals),
        }
    
    def _check_fib_interaction(self, high: float, low: float, close: float,
                               basis: float, upper_618: float, lower_618: float,
                               upper_1: float, lower_1: float) -> Dict:
        """Check if candle interacts with Fibonacci levels"""
        tolerance = (upper_1 - lower_1) * 0.02  # 2% tolerance band
        
        interactions = []
        
        # Check each level
        levels = {
            'basis': basis,
            'upper_618': upper_618,
            'lower_618': lower_618,
            'upper_1': upper_1,
            'lower_1': lower_1,
        }
        
        for level_name, level_price in levels.items():
            if low <= level_price <= high:  # Level is within candle range
                interactions.append({
                    'level': level_name,
                    'price': level_price,
                    'interaction_type': 'touch',
                })
            elif abs(close - level_price) < tolerance:  # Close is near level
                interactions.append({
                    'level': level_name,
                    'price': level_price,
                    'interaction_type': 'near',
                    'distance': abs(close - level_price),
                })
        
        return {'levels': interactions, 'has_interaction': len(interactions) > 0}
    
    def _find_nearby_obs(self, current_index: int, ob_result: Dict, lookback: int = 50) -> List[Dict]:
        """Find Order Blocks within lookback range"""
        nearby = []
        
        for ob in ob_result['all']:
            if current_index - lookback <= ob['index'] <= current_index + 20:
                nearby.append(ob)
        
        return nearby
    
    def _calculate_signal_strength(self, current_index: int, fib_interaction: Dict,
                                   nearby_obs: List[Dict], candle_close: float,
                                   basis: float) -> float:
        """Calculate fusion signal strength (0-100)"""
        strength = 0.0
        
        # Factor 1: Number of Fibonacci levels interacting
        interaction_count = len(fib_interaction['levels'])
        strength += min(interaction_count * 20, 30)
        
        # Factor 2: Number of nearby Order Blocks
        ob_count = len(nearby_obs)
        strength += min(ob_count * 15, 30)
        
        # Factor 3: OB direction alignment
        bullish_obs = sum(1 for ob in nearby_obs if ob['type'] == 'bullish')
        bearish_obs = sum(1 for ob in nearby_obs if ob['type'] == 'bearish')
        
        if candle_close < basis and bearish_obs > 0:
            strength += 20  # Bearish alignment
        elif candle_close > basis and bullish_obs > 0:
            strength += 20  # Bullish alignment
        
        # Factor 4: Proximity to OB
        if nearby_obs:
            closest_ob = min(nearby_obs, key=lambda ob: abs(ob['avg'] - candle_close))
            distance_pct = abs(closest_ob['avg'] - candle_close) / candle_close * 100
            if distance_pct < 1:
                strength += 20
            elif distance_pct < 3:
                strength += 10
        
        return min(strength, 100)


if __name__ == '__main__':
    # Test example
    sample_data = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 102,
        'low': np.random.randn(500).cumsum() + 98,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500),
    })
    
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + abs(np.random.randn(500))
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - abs(np.random.randn(500))
    
    fusion = FibOBFusion(fib_length=200, fib_mult=3.0, ob_periods=5)
    result = fusion.analyze(sample_data)
    
    print(f"Fibonacci Basis calculated: {len(result['fib']['basis'])} values")
    print(f"Bullish Order Blocks: {len(result['ob']['bullish'])}")
    print(f"Bearish Order Blocks: {len(result['ob']['bearish'])}")
    print(f"Fusion Signals: {result['relationships']['signal_count']}")
