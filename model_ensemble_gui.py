import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os


class SupplyDemandDetector:
    def __init__(self, aggregation_factor=15, zone_length=100, 
                 show_supply_zones=True, show_demand_zones=True):
        self.aggregation_factor = aggregation_factor
        self.zone_length = zone_length
        self.show_supply_zones = show_supply_zones
        self.show_demand_zones = show_demand_zones

    def detect_zones(self, df: pd.DataFrame):
        """
        偵測 Supply / Demand 區域
        
        核心邏輯：
        1. 聚合 K 線：以 aggregation_factor 根 bar 為單位聚合
        2. 方向轉變判斷：當聚合 K 線方向改變時（bullish->bearish 或 bearish->bullish）
        3. 只在轉變點生成 zone，防止密集生成
        4. demand zone：在 bearish->bullish 轉變時生成
        5. supply zone：在 bullish->bearish 轉變時生成
        """
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)

        # 聚合組狀態
        group_start_idx = None
        agg_open = agg_high = agg_low = agg_close = None
        prev_is_bullish = None  # 追蹤上一個聚合 K 線是否為 bullish

        # 前一個 zone 的資訊
        prev_supply_low = None
        prev_supply_high = None
        prev_supply_start = None

        prev_demand_low = None
        prev_demand_high = None
        prev_demand_start = None

        # Zone 使用標誌
        supply_zone_used = False
        demand_zone_used = False

        supply_zones = []
        demand_zones = []

        for i in range(n):
            # 初始化第一組
            if group_start_idx is None:
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]
                continue

            # 聚合更新
            agg_high = max(agg_high, h[i])
            agg_low = min(agg_low, l[i])
            agg_close = c[i]

            bars_in_group = i - group_start_idx + 1
            is_new_group = bars_in_group >= self.aggregation_factor

            if is_new_group:
                # 聚合 K 線完成
                is_bullish = agg_close >= agg_open

                # 方向改變判斷
                if prev_is_bullish is not None:
                    if is_bullish and not prev_is_bullish:
                        # bearish -> bullish 轉變
                        supply_zone_used = False  # 重設供應區使用標誌
                        
                        # 產生 demand zone（從前一個 bearish 週期）
                        if prev_demand_high is not None and not demand_zone_used and self.show_demand_zones:
                            zone_start = prev_demand_start
                            zone_end = min(prev_demand_start + self.zone_length, n - 1)
                            demand_zones.append({
                                'start_idx': zone_start,
                                'end_idx': zone_end,
                                'high': prev_demand_high,
                                'low': prev_demand_low,
                            })
                            demand_zone_used = True

                    elif not is_bullish and prev_is_bullish:
                        # bullish -> bearish 轉變
                        demand_zone_used = False  # 重設需求區使用標誌
                        
                        # 產生 supply zone（從前一個 bullish 週期）
                        if prev_supply_low is not None and not supply_zone_used and self.show_supply_zones:
                            zone_start = prev_supply_start
                            zone_end = min(prev_supply_start + self.zone_length, n - 1)
                            supply_zones.append({
                                'start_idx': zone_start,
                                'end_idx': zone_end,
                                'high': prev_supply_high,
                                'low': prev_supply_low,
                            })
                            supply_zone_used = True

                # 記錄當前週期為下次使用
                if is_bullish:
                    prev_demand_low = agg_low
                    prev_demand_high = agg_high
                    prev_demand_start = group_start_idx
                else:
                    prev_supply_low = agg_low
                    prev_supply_high = agg_high
                    prev_supply_start = group_start_idx

                # 追蹤目前方向
                prev_is_bullish = is_bullish
                
                # 新聚合組開始
                group_start_idx = i
                agg_open = o[i]
                agg_high = h[i]
                agg_low = l[i]
                agg_close = c[i]

        return {
            'supply': supply_zones,
            'demand': demand_zones,
        }


def main():
    print("Supply/Demand Zone Detector Test")
    print("Module loaded successfully")


if __name__ == '__main__':
    main()
