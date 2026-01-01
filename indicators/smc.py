import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Structure(Enum):
    """結構類型定義"""
    BULLISH = 1  # 看漲結構
    BEARISH = -1  # 看跌結構


@dataclass
class Zone:
    """Supply/Demand Zone 定義"""
    bar_index: int  # 生成的 K 線索引
    high: float
    low: float
    structure_type: Structure
    created_at_idx: int  # 實際創建位置索引

    @property
    def is_supply(self) -> bool:
        return self.structure_type == Structure.BEARISH

    @property
    def is_demand(self) -> bool:
        return self.structure_type == Structure.BULLISH

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    def contains_price(self, price: float) -> bool:
        return self.low <= price <= self.high


@dataclass
class Pivot:
    """樞紐點定義"""
    bar_index: int
    price: float
    is_higher_high: bool  # 是否為更高的高點
    is_lower_low: bool  # 是否為更低的低點


@dataclass
class Leg:
    """腿部定義 - 連續的相同方向移動"""
    bar_index: int
    high: float
    low: float
    direction: Structure  # BULLISH = 上升, BEARISH = 下跌
    length: int  # 腿部包含的 K 線數量


class SmartMoneyStructure:
    """Smart Money Concept 結構分析"""

    def __init__(self, df: pd.DataFrame, pivot_lookback: int = 5, min_leg_length: int = 3):
        """
        初始化 SMC 分析器
        
        Args:
            df: 包含 'high', 'low', 'close' 的 K 線 DataFrame
            pivot_lookback: 樞紐點計算回看周期
            min_leg_length: 最小腿部長度
        """
        self.df = df.reset_index(drop=True)
        self.pivot_lookback = pivot_lookback
        self.min_leg_length = min_leg_length
        
        # 輸出存儲
        self.legs: List[Leg] = []
        self.pivots: List[Pivot] = []
        self.zones: List[Zone] = []
        
        # 內部狀態
        self._prev_is_bullish: Optional[bool] = None
        self._last_zone_idx: Optional[int] = None
        self._supply_zone_used = False
        self._demand_zone_used = False
        
    def analyze(self) -> None:
        """執行完整的 SMC 分析"""
        self._identify_legs()
        self._identify_pivots()
        self._generate_zones()

    def _identify_legs(self) -> None:
        """識別 SMC 腿部 - 連續的相同方向移動"""
        self.legs = []
        i = 1
        
        while i < len(self.df):
            prev_high = self.df.loc[i - 1, 'high']
            prev_low = self.df.loc[i - 1, 'low']
            curr_high = self.df.loc[i, 'high']
            curr_low = self.df.loc[i, 'low']
            
            # 判斷上升或下跌
            is_higher = curr_high > prev_high and curr_low > prev_low
            is_lower = curr_high < prev_high and curr_low < prev_low
            
            if not (is_higher or is_lower):
                i += 1
                continue
            
            # 確定當前腿部方向
            direction = Structure.BULLISH if is_higher else Structure.BEARISH
            leg_start = i - 1
            leg_high = max(prev_high, curr_high)
            leg_low = min(prev_low, curr_low)
            leg_length = 2
            
            # 延長腿部
            i += 1
            while i < len(self.df):
                next_high = self.df.loc[i, 'high']
                next_low = self.df.loc[i, 'low']
                
                if direction == Structure.BULLISH:
                    if next_high > leg_high and next_low > leg_low:
                        leg_high = next_high
                        leg_low = next_low
                        leg_length += 1
                        i += 1
                    else:
                        break
                else:  # BEARISH
                    if next_high < leg_high and next_low < leg_low:
                        leg_high = next_high
                        leg_low = next_low
                        leg_length += 1
                        i += 1
                    else:
                        break
            
            if leg_length >= self.min_leg_length:
                self.legs.append(Leg(
                    bar_index=leg_start,
                    high=leg_high,
                    low=leg_low,
                    direction=direction,
                    length=leg_length
                ))
            
            i += 1

    def _identify_pivots(self) -> None:
        """識別樞紐點 (Pivot Points)"""
        self.pivots = []
        
        for i in range(self.pivot_lookback, len(self.df) - self.pivot_lookback):
            # 更高的高點 (Higher High)
            if (self.df.loc[i, 'high'] > self.df.loc[i - self.pivot_lookback:i, 'high'].max() and
                self.df.loc[i, 'high'] > self.df.loc[i:i + self.pivot_lookback + 1, 'high'].max()):
                self.pivots.append(Pivot(
                    bar_index=i,
                    price=self.df.loc[i, 'high'],
                    is_higher_high=True,
                    is_lower_low=False
                ))
            
            # 更低的低點 (Lower Low)
            if (self.df.loc[i, 'low'] < self.df.loc[i - self.pivot_lookback:i, 'low'].min() and
                self.df.loc[i, 'low'] < self.df.loc[i:i + self.pivot_lookback + 1, 'low'].min()):
                self.pivots.append(Pivot(
                    bar_index=i,
                    price=self.df.loc[i, 'low'],
                    is_higher_high=False,
                    is_lower_low=True
                ))

    def _generate_zones(self) -> None:
        """生成 Supply/Demand Zone
        
        核心邏輯改進:
        1. 追蹤上一個聚合 K 線的方向 (prev_is_bullish)
        2. 只在方向轉換時生成 zone
        3. 防止連續生成重複 zone
        """
        self.zones = []
        self._prev_is_bullish = None
        self._supply_zone_used = False
        self._demand_zone_used = False
        
        for i in range(1, len(self.df)):
            curr_close = self.df.loc[i, 'close']
            curr_high = self.df.loc[i, 'high']
            curr_low = self.df.loc[i, 'low']
            prev_high = self.df.loc[i - 1, 'high']
            prev_low = self.df.loc[i - 1, 'low']
            
            # 判斷當前 K 線的方向
            is_bullish = curr_close >= (curr_high + curr_low) / 2
            
            if self._prev_is_bullish is None:
                self._prev_is_bullish = is_bullish
                continue
            
            # 方向轉換判斷
            direction_changed = self._prev_is_bullish != is_bullish
            
            if direction_changed:
                if self._prev_is_bullish:  # 從 bullish 轉向 bearish -> 生成 Supply Zone
                    if not self._supply_zone_used:
                        zone = Zone(
                            bar_index=i - 1,
                            high=prev_high,
                            low=prev_low,
                            structure_type=Structure.BEARISH,
                            created_at_idx=i - 1
                        )
                        self.zones.append(zone)
                        self._supply_zone_used = True
                        self._demand_zone_used = False  # 重設需求 zone 狀態
                
                else:  # 從 bearish 轉向 bullish -> 生成 Demand Zone
                    if not self._demand_zone_used:
                        zone = Zone(
                            bar_index=i - 1,
                            high=prev_high,
                            low=prev_low,
                            structure_type=Structure.BULLISH,
                            created_at_idx=i - 1
                        )
                        self.zones.append(zone)
                        self._demand_zone_used = True
                        self._supply_zone_used = False  # 重設供給 zone 狀態
                
                self._prev_is_bullish = is_bullish
            else:
                self._prev_is_bullish = is_bullish

    def get_zones_df(self) -> pd.DataFrame:
        """以 DataFrame 格式返回 zones"""
        if not self.zones:
            return pd.DataFrame()
        
        data = []
        for zone in self.zones:
            data.append({
                'bar_index': zone.bar_index,
                'high': zone.high,
                'low': zone.low,
                'mid': zone.mid,
                'type': 'Supply' if zone.is_supply else 'Demand',
                'created_at_idx': zone.created_at_idx
            })
        
        return pd.DataFrame(data)

    def get_closest_zone(self, current_price: float, max_distance: Optional[float] = None) -> Optional[Zone]:
        """取得最接近當前價格的 zone
        
        Args:
            current_price: 當前價格
            max_distance: 最大距離限制 (百分比)
        """
        closest = None
        min_distance = float('inf')
        
        for zone in self.zones:
            if zone.contains_price(current_price):
                return zone
            
            # 計算到 zone 的距離
            distance = min(abs(current_price - zone.high), abs(current_price - zone.low))
            
            if max_distance:
                threshold = current_price * (max_distance / 100)
                if distance > threshold:
                    continue
            
            if distance < min_distance:
                min_distance = distance
                closest = zone
        
        return closest

    def get_broken_zones(self, current_price: float) -> List[Zone]:
        """取得已被突破的 zones (價格穿過 zone)"""
        broken = []
        for zone in self.zones:
            if zone.is_supply and current_price < zone.low:
                broken.append(zone)
            elif zone.is_demand and current_price > zone.high:
                broken.append(zone)
        
        return broken

    def get_active_zones(self, current_price: float, tolerance: float = 0.01) -> List[Zone]:
        """取得活躍的 zones (價格在 zone 附近)
        
        Args:
            current_price: 當前價格
            tolerance: 容差 (百分比)
        """
        active = []
        threshold = current_price * (tolerance / 100)
        
        for zone in self.zones:
            if (zone.low - threshold <= current_price <= zone.high + threshold):
                active.append(zone)
        
        return active
