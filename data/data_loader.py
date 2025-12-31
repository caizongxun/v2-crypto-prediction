"""
數據加載器 - 從 HuggingFace 加載 BTC 15分鐘時間框架數據
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from huggingface_hub import HfApi
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    數據加載器
    """
    
    def __init__(self, hf_token: str, repo_id: str = "zongowo111/v2-crypto-ohlcv-data"):
        """
        初始化數據加載器
        
        Args:
            hf_token: HuggingFace API token
            repo_id: HuggingFace 資料庫 ID
        """
        self.api = HfApi(token=hf_token)
        self.repo_id = repo_id
        self.repo_type = "dataset"
        self.base_path = "klines"
        self.data_cache = {}
        
    def list_available_pairs(self) -> list:
        """
        列出所有可用的交易對
        
        Returns:
            list: 交易對名稱清單
        """
        try:
            repo_info = self.api.repo_info(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                files_metadata=True
            )
            
            pairs = set()
            
            if hasattr(repo_info, 'siblings'):
                for sibling in repo_info.siblings:
                    file_path = sibling.rfilename
                    
                    if file_path.startswith(self.base_path + "/"):
                        parts = file_path.split("/")
                        
                        if len(parts) >= 2:
                            pair_name = parts[1]
                            if pair_name.upper().endswith("USDT"):
                                pairs.add(pair_name)
            
            return sorted(list(pairs))
            
        except Exception as e:
            logger.error(f"列出交易對失敗: {e}")
            return []
    
    def load_pair_data(
        self,
        pair: str,
        local_cache_dir: str = ".cache"
    ) -> Optional[pd.DataFrame]:
        """
        加載批串數據
        
        Args:
            pair: 交易對 (如 "BTCUSDT")
            local_cache_dir: 本地快取目錄
        
        Returns:
            Optional[pd.DataFrame]: K線數據
        """
        try:
            logger.info(f"加載 {pair} 數據...")
            
            # 創建快取目錄
            os.makedirs(local_cache_dir, exist_ok=True)
            
            pair_path = f"{self.base_path}/{pair}"
            
            # 列出該交易對的所有檔案
            repo_info = self.api.repo_info(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                files_metadata=True
            )
            
            data_files = []
            if hasattr(repo_info, 'siblings'):
                for sibling in repo_info.siblings:
                    file_path = sibling.rfilename
                    
                    if file_path.startswith(pair_path + "/"):
                        if file_path.endswith(".csv"):
                            data_files.append(file_path)
            
            if not data_files:
                logger.warning(f"找不到 {pair} 的數據檔案")
                return None
            
            # 加載並合併的數據
            dfs = []
            
            for file_path in sorted(data_files):
                local_file = os.path.join(local_cache_dir, os.path.basename(file_path))
                
                # 從 HuggingFace 下載
                file_url = self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    filename=file_path,
                    cache_dir=local_cache_dir,
                    force_download=False
                )
                
                try:
                    df = pd.read_csv(file_url)
                    dfs.append(df)
                    logger.info(f"  加載: {os.path.basename(file_path)} ({len(df)} 行)")
                except Exception as e:
                    logger.warning(f"  加載失敗: {file_path} - {e}")
            
            if not dfs:
                logger.warning(f"無法加載 {pair} 的任何數據")
                return None
            
            # 合併所有 dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # 排序並移除重複
            combined_df = combined_df.sort_values("open_time", ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["open_time"], keep='first')
            
            # 轉換時間爲 datetime
            if "open_time" in combined_df.columns:
                combined_df["open_time"] = pd.to_datetime(combined_df["open_time"], unit='ms')
            
            logger.info(f"{pair} 加載完成: {len(combined_df)} 行數據")
            
            # 快取中間結果
            self.data_cache[pair] = combined_df
            
            return combined_df
            
        except Exception as e:
            logger.error(f"加載 {pair} 數據失敗: {e}")
            return None
    
    def get_cached_data(self, pair: str) -> Optional[pd.DataFrame]:
        """
        取得快取的數據
        
        Args:
            pair: 交易對
        
        Returns:
            Optional[pd.DataFrame]: 快取數據
        """
        return self.data_cache.get(pair)
    
    def clear_cache(self):
        """清除快取"""
        self.data_cache.clear()
        logger.info("快取已清除")


def load_btc_data(
    hf_token: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    快速加載 BTC 數據
    
    Args:
        hf_token: HuggingFace token
        start_date: 開始日期 (可選)
        end_date: 結束日期 (可選)
    
    Returns:
        Optional[pd.DataFrame]: BTC OHLCV 數據
    """
    loader = DataLoader(hf_token)
    df = loader.load_pair_data("BTCUSDT")
    
    if df is None:
        return None
    
    # 時間篆入
    if start_date:
        df = df[df["open_time"] >= start_date]
    
    if end_date:
        df = df[df["open_time"] <= end_date]
    
    return df
