"""
配置文件 - 定義常數和設定
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# 加載 .env 檔案
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# HuggingFace 設定
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
HF_REPO_ID: str = "zongowo111/v2-crypto-ohlcv-data"
HF_REPO_TYPE: str = "dataset"

# 資料夾設定
BASE_PATH: str = "klines"
REQUIRED_SUFFIX: str = "USDT"

# 交易對設定
TRADING_PAIR: str = os.getenv("TRADING_PAIR", "BTCUSDT")
TIMEFRAME: str = os.getenv("TIMEFRAME", "15m")

# 資料設定
DATA_PATH: str = "data"
MODEL_PATH: str = "models"
RESULTS_PATH: str = "results"

# 回測設定
START_DATE: str = os.getenv("START_DATE", "2023-01-01")
END_DATE: str = os.getenv("END_DATE", "2025-12-31")

# 日誌設定
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = "prediction.log"

# K線設定
CANDLE_COLUMNS: list = ["open", "high", "low", "close", "volume"]

# 時間相關
HOURS_PER_DAY: int = 24
CANDLES_PER_HOUR: int = 4  # 15分鐘 K線

# 公式設定
LOOKBACK_PERIOD: int = int(os.getenv("LOOKBACK_PERIOD", "20"))
MIN_PATTERN_STRENGTH: float = float(os.getenv("MIN_PATTERN_STRENGTH", "0.7"))

def validate_config() -> bool:
    """驗證必要的配置"""
    if not HF_TOKEN:
        print("警告: HF_TOKEN 環境變數未設定")
        print("請在 .env 檔案中設定 HF_TOKEN")
        return False
    
    if not HF_REPO_ID:
        print("錯誤: HF_REPO_ID 未設定")
        return False
    
    return True

def print_config():
    """列印當前配置"""
    print("\n" + "=" * 70)
    print("當前配置")
    print("=" * 70)
    print(f"HF_TOKEN: {'設定' if HF_TOKEN else '未設定'}")
    print(f"交易對: {TRADING_PAIR}")
    print(f"時間框架: {TIMEFRAME}")
    print(f"回測期間: {START_DATE} 至 {END_DATE}")
    print(f"回看週期: {LOOKBACK_PERIOD}")
    print(f"最小樣式強度: {MIN_PATTERN_STRENGTH}")
    print(f"日誌級別: {LOG_LEVEL}")
    print("=" * 70 + "\n")
