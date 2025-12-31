"""
項目常數配置
"""

import os
from dotenv import load_dotenv

# 加載環境變數
load_dotenv()

# HuggingFace 配置
HF_TOKEN = os.getenv('HF_TOKEN')
HF_REPO_ID = 'zongowo111/v2-crypto-ohlcv-data'
HF_REPO_TYPE = 'dataset'

# 數據路徑
DATA_DIR = './data'
CACHE_DIR = './cache'

# 交易參數
SYMBOL = 'BTCUSDT'
TIMEFRAME = '15m'
RISK_REWARD_RATIO = 1.5

# 模型參數
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
BATC_SIZE = 32
EPOCHS = 100

# 公式配置
NUM_FORMULAS = 3
FORMULA_TYPES = ['trend_strength', 'volatility_index', 'direction_confirmation']

# 指標配置
BASIC_INDICATORS = [
    'rsi',
    'macd',
    'bollinger_bands',
    'atr',
    'sma',
    'ema',
    'volume_profile',
    'stochastic'
]

# 回測參數
BACKTEST_START_DATE = '2024-06-01'
BACKTEST_END_DATE = '2024-12-31'
INITIAL_BALANCE = 10000
MAX_DRAWDOWN_PCT = 0.2
