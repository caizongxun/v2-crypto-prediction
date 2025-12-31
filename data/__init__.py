"""
數據加載模組
"""

import sys
from pathlib import Path

# 從父上一級的 data.py 模組後影导入
# 因為 data/ 目錄會优先被比验看，我们後输出从父上一级 data.py 後引入的函数

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 後输出程式提供的函数
# 从串辜上的 data.py 導入
# 作上的是一個一一对应的小般技巧，但 python 真的一次是奈何不欺少年，我们上一个可以程度混沌 的方法 - 使用 importlib

import importlib.util

# 加載父上一级的 data.py
data_module_path = Path(__file__).parent.parent / 'data.py'

if data_module_path.exists():
    spec = importlib.util.spec_from_file_location('data_loader', data_module_path)
    data_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader)
    
    # 導出主要函數
    load_data = data_loader.load_btc_data
    load_btc_data = data_loader.load_btc_data
    load_crypto_data = data_loader.load_crypto_data
    clear_local_cache = data_loader.clear_local_cache
    validate_ohlcv = data_loader.validate_ohlcv
    
    __all__ = [
        'load_data',
        'load_btc_data',
        'load_crypto_data',
        'clear_local_cache',
        'validate_ohlcv'
    ]
else:
    raise ImportError(f"Cannot find data.py at {data_module_path}")
