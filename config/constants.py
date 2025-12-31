"""
配置文件 - 定義常數和設定
"""

import os
from typing import Optional

# HuggingFace 設定
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
HF_REPO_ID: str = "zongowo111/v2-crypto-ohlcv-data"
HF_REPO_TYPE: str = "dataset"

# 資料夾設定
BASE_PATH: str = "klines"
REQUIRED_SUFFIX: str = "USDT"

# 交易對設定
TRADING_PAIR: str = "BTCUSDT"
TIMEFRAME: str = "15m"

# 資料設定
DATA_PATH: str = "data"
MODEL_PATH: str = "models"
RESULTS_PATH: str = "results"

# 回測設定
START_DATE: str = "2023-01-01"
END_DATE: str = "2025-12-31"

# 日誌設定
LOG_LEVEL: str = "INFO"
LOG_FILE: str = "prediction.log"

# K線設定
CANDLE_COLUMNS: list = ["open", "high", "low", "close", "volume"]

# 時間相關
HOURS_PER_DAY: int = 24
CANDLES_PER_HOUR: int = 4  # 15分鐘 K線

def validate_config() -> bool:
    """驗證必要的配置"""
    if not HF_TOKEN:
        print("警告: HF_TOKEN 環境變數未設定")
        return False
    
    if not HF_REPO_ID:
        print("錯誤: HF_REPO_ID 未設定")
        return False
    
    return True
