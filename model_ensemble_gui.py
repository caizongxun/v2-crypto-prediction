    def plot_kline_with_zones(self, df: pd.DataFrame, zones: dict, display_bars: int = 1000):
        """繪製 K 線與 Supply/Demand 區域
        
        Args:
            df: 完整的 K 線 DataFrame
            zones: Supply/Demand zones 字典
            display_bars: 要顯示的最近 K 棒數 (預設 1000)
        """
        self.ax.clear()

        if df is None or len(df) == 0:
            self.ax.text(0.5, 0.5, "尚未載入數據", ha='center', va='center', transform=self.ax.transAxes)
            self.draw()
            return

        # 決定要顯示的範圍（最後 display_bars 根）
        total_bars = len(df)
        start_idx = max(0, total_bars - display_bars)
        end_idx = total_bars
        
        # 提取要顯示的數據
        df_display = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # 繪製基本 K 線 (簡化版 OHLC 條)
        x = np.arange(len(df_display))
        o = df_display['open'].values
        h = df_display['high'].values
        l = df_display['low'].values
        c = df_display['close'].values

        # 上漲 K 線
        up = c >= o
        down = ~up

        # 畫上下影線
        self.ax.vlines(x, l, h, color='black', linewidth=0.5, alpha=0.6)
        # 實體
        self.ax.vlines(x[up], o[up], c[up], color='green', linewidth=4)
        self.ax.vlines(x[down], c[down], o[down], color='red', linewidth=4)

        # 繪製 Supply/Demand Zones
        if zones is not None:
            # Supply Zones
            for z in zones.get('supply', []):
                # 檢查該 zone 是否在顯示範圍內
                zone_start = z['start_idx']
                zone_end = z['end_idx']
                
                # 將原始索引轉換為顯示範圍內的索引
                if zone_end < start_idx or zone_start >= end_idx:
                    # Zone 完全在顯示範圍外，跳過
                    continue
                
                # 計算實際要顯示的起點和終點（在顯示範圍內）
                display_zone_start = max(zone_start - start_idx, 0)
                display_zone_end = min(zone_end - start_idx, len(df_display) - 1)
                
                if display_zone_start < len(df_display):
                    xmin = display_zone_start / len(df_display)
                    xmax = display_zone_end / len(df_display)
                    self.ax.axhspan(z['low'], z['high'], xmin=xmin, xmax=xmax,
                                    facecolor=(1, 0, 0, 0.2), edgecolor='red', linewidth=1)
            
            # Demand Zones
            for z in zones.get('demand', []):
                # 檢查該 zone 是否在顯示範圍內
                zone_start = z['start_idx']
                zone_end = z['end_idx']
                
                if zone_end < start_idx or zone_start >= end_idx:
                    # Zone 完全在顯示範圍外，跳過
                    continue
                
                # 計算實際要顯示的起點和終點（在顯示範圍內）
                display_zone_start = max(zone_start - start_idx, 0)
                display_zone_end = min(zone_end - start_idx, len(df_display) - 1)
                
                if display_zone_start < len(df_display):
                    xmin = display_zone_start / len(df_display)
                    xmax = display_zone_end / len(df_display)
                    self.ax.axhspan(z['low'], z['high'], xmin=xmin, xmax=xmax,
                                    facecolor=(0, 1, 0, 0.2), edgecolor='green', linewidth=1)

        self.ax.set_title(f"K 線與 Supply/Demand 區域 (最近 {len(df_display)} 根, 索引 {start_idx}-{end_idx})")
        self.ax.set_xlabel("Bar Index (相對於顯示範圍)")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, alpha=0.2)
        self.fig.tight_layout()
        self.draw()