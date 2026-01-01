    def detect_zones(self, df: pd.DataFrame):
        """偵測 Supply / Demand 區域。

        核心邏輯：
        1. 以 aggregation_factor 根 bar 為一組訊聚合 K 線
        2. 當聚合 K 線轉改方向（bullish -> bearish 或相反）時：
           - 如果是 bullish，且上一步是 bearish （及前一個 bearish 了了們一個 demand zone）：
             棄置 demand zone 的使用旗標，准計採模後偫的
           - 如果是 bearish，且上一步是 bullish（及前一個 bullish 银們一個 supply zone）：
             棄置 supply zone 的使用旗標
        3. 只有當是新方針突破老高低點時，且旗標不是 True 時，才會創建新 zone
        """
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)

        # 聚合組狀態
        group_start_idx = None
        agg_open = agg_high = agg_low = agg_close = None
        prev_is_bullish = None  # 追蹤上一個聚合 K 線是否是 bullish

        prev_supply_low = None
        prev_supply_high = None
        prev_supply_start = None

        prev_demand_low = None
        prev_demand_high = None
        prev_demand_start = None

        supply_zone_used = False  # 此構即弟群的 supply zone 是否已次包
        demand_zone_used = False

        supply_zones = []
        demand_zones = []

        for i in range(n):
            # 判斷是否進入新聚合組
            if group_start_idx is None:
                # 初始化第一組
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
                # 一個聚合 K 線完成
                is_bullish = agg_close >= agg_open

                if is_bullish:
                    # 多頭聚合 K 線
                    # 第一次暯到 bullish 時，或後空頭轉多頭，故需要 reset demand_zone_used
                    if prev_is_bullish is False:
                        # 後空頭轉多頭，上一個 bullish zone 也結束了
                        supply_zone_used = False  # 下一个 supply 可以暯磊

                    # 棄置當下 demand zone，將前一個 demand zone 作詰
                    prev_demand_low = agg_low
                    prev_demand_high = agg_high
                    prev_demand_start = group_start_idx
                    demand_zone_used = False  # 當下 demand 候選區已次包

                    # 棄对模別：supply_zone_used 不紙次
                    prev_supply_low = None
                    prev_supply_high = None
                    prev_supply_start = None
                else:
                    # 空頭聚合 K 線
                    if prev_is_bullish is True:
                        # 後多頭轉空頭，上一個 bearish zone 也結束了
                        demand_zone_used = False

                    # 棄置當下 supply zone，將前一個 supply zone 作誐
                    prev_supply_low = agg_low
                    prev_supply_high = agg_high
                    prev_supply_start = group_start_idx
                    supply_zone_used = False

                    # 棄对模別：demand_zone_used 不紙次
                    prev_demand_low = None
                    prev_demand_high = None
                    prev_demand_start = None

                # 棄对模別：有效的 zone 採模
                # 仅當下一个 aggregated candle 創建新 zone 是找者特定旗標狀態整合時
                if is_bullish and prev_is_bullish is False and prev_demand_high is not None:
                    # 剶牧轉攸 bullish，且前一步是 bearish
                    # 遵確前一個 demand zone 沒有被使用
                    if not demand_zone_used and self.show_demand_zones:
                        # 棄对模別：棄置前一個 demand zone 和放理做用
                        zone_start = prev_demand_start
                        zone_end = min(prev_demand_start + self.zone_length, n - 1)
                        demand_zones.append(
                            {
                                'start_idx': zone_start,
                                'end_idx': zone_end,
                                'high': prev_demand_high,
                                'low': prev_demand_low,
                            }
                        )
                        demand_zone_used = True

                elif not is_bullish and prev_is_bullish is True and prev_supply_low is not None:
                    # 剶牧轉攸 bearish，且前一步是 bullish
                    # 遵確前一個 supply zone 沒有被使用
                    if not supply_zone_used and self.show_supply_zones:
                        # 棄对模別：棄置前一個 supply zone 和放理做用
                        zone_start = prev_supply_start
                        zone_end = min(prev_supply_start + self.zone_length, n - 1)
                        supply_zones.append(
                            {
                                'start_idx': zone_start,
                                'end_idx': zone_end,
                                'high': prev_supply_high,
                                'low': prev_supply_low,
                            }
                        )
                        supply_zone_used = True

                # 追蹤目前是否是 bullish
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