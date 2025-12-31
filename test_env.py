#!/usr/bin/env python3
"""
測試 .env 配置是否正確加載
"""

from config import (
    HF_TOKEN,
    TRADING_PAIR,
    TIMEFRAME,
    START_DATE,
    END_DATE,
    LOOKBACK_PERIOD,
    MIN_PATTERN_STRENGTH,
    LOG_LEVEL,
    print_config,
    validate_config
)

def main():
    print("\n測試 .env 檔案配置...")
    print("=" * 70)
    
    # 列印所有配置
    print_config()
    
    # 驗證配置
    if validate_config():
        print("配置驗證: 成功")
        print(f"  HF_TOKEN: 已設定")
        print(f"  TRADING_PAIR: {TRADING_PAIR}")
        print(f"  TIMEFRAME: {TIMEFRAME}")
        print(f"  START_DATE: {START_DATE}")
        print(f"  END_DATE: {END_DATE}")
        print(f"  LOOKBACK_PERIOD: {LOOKBACK_PERIOD}")
        print(f"  MIN_PATTERN_STRENGTH: {MIN_PATTERN_STRENGTH}")
        print(f"  LOG_LEVEL: {LOG_LEVEL}")
        print("\n所有配置成功加載！")
        return 0
    else:
        print("配置驗證: 失敗")
        return 1

if __name__ == "__main__":
    exit(main())
