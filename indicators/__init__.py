"""
基礎技術指標實現
"""

from .rsi import calculate_rsi
from .macd import calculate_macd
from .bollinger_bands import calculate_bollinger_bands
from .atr import calculate_atr
from .moving_averages import calculate_sma, calculate_ema
from .volume_profile import calculate_volume_sma
from .stochastic import calculate_stochastic


class IndicatorCalculator:
    """
    技術指標計算器
    一次性計算所有指標
    """
    
    def __init__(self):
        pass
    
    def calculate_all(self, df):
        """
        計算所有技術指標
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            dict: {indicator_name: values}
        """
        indicators = {}
        
        # RSI
        indicators['rsi'] = calculate_rsi(df['close'])
        
        # MACD
        macd_result = calculate_macd(df['close'])
        if isinstance(macd_result, tuple):
            indicators['macd_line'], indicators['signal_line'], indicators['histogram'] = macd_result
        else:
            indicators['macd_line'] = macd_result
            indicators['signal_line'] = None
            indicators['histogram'] = None
        
        # Bollinger Bands
        bb_result = calculate_bollinger_bands(df['close'])
        if isinstance(bb_result, tuple) and len(bb_result) == 3:
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = bb_result
        else:
            indicators['bb_upper'] = None
            indicators['bb_middle'] = None
            indicators['bb_lower'] = None
        
        # ATR
        indicators['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        
        # Moving Averages
        indicators['sma_20'] = calculate_sma(df['close'], 20)
        indicators['ema_12'] = calculate_ema(df['close'], 12)
        indicators['ema_26'] = calculate_ema(df['close'], 26)
        
        # Volume
        indicators['volume_sma'] = calculate_volume_sma(df['volume'])
        
        # Stochastic
        stoch_result = calculate_stochastic(df['high'], df['low'], df['close'])
        if isinstance(stoch_result, tuple) and len(stoch_result) == 2:
            indicators['stochastic_k'], indicators['stochastic_d'] = stoch_result
        else:
            indicators['stochastic_k'] = None
            indicators['stochastic_d'] = None
        
        return indicators


__all__ = [
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_atr',
    'calculate_sma',
    'calculate_ema',
    'calculate_volume_sma',
    'calculate_stochastic',
    'IndicatorCalculator',
]
