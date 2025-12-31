"""
回測引擎 - 計算交易表現

目稱函數可以是：
- Sharpe Ratio
- 勝率
- 收益率
- 最大回撤
- 期望值
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class PositionType(Enum):
    """位置類別"""
    LONG = 1  # 幻趥
    SHORT = -1  # 窗趥
    NEUTRAL = 0  # 中性


@dataclass
class Trade:
    """单筆买賣"""
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    position_type: PositionType
    pnl: float  # 損益
    pnl_percent: float  # 損益率
    confidence: float  # 信忆稿


@dataclass
class BacktestResult:
    """回測結果"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    total_return: float  # 總抠益 (%)
    annual_return: float  # 年化抠益 (%)
    sharpe_ratio: float  # Sharpe 比
    max_drawdown: float  # 最大回撤 (%)
    
    avg_win: float  # 平均获利
    avg_loss: float  # 平均乿損
    profit_factor: float  # 装市系数 (profit/loss)
    
    trades: List[Trade]
    equity_curve: pd.Series


class BacktestEngine:
    """
    回測引擎
    """
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        """
        初始化回測引擎
        
        Args:
            initial_capital: 初始詳治 (預設 10000)
            commission: 手續費 (預設 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades: List[Trade] = []
        self.equity_curve = []
    
    def run(
        self,
        df: pd.DataFrame,
        signals: List[Tuple[int, int]]  # [(index, signal)]
    ) -> BacktestResult:
        """
        執行回測
        
        Args:
            df: OHLCV 数据（需要有 'close' 列）
            signals: 信號列表 [(index, signal)]
                    signal: 1=BUY, -1=SELL, 0=NEUTRAL
        
        Returns:
            BacktestResult: 回測結果
        """
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        current_position = PositionType.NEUTRAL
        entry_price = 0
        entry_index = 0
        entry_confidence = 0
        
        for signal_index, (index, signal) in enumerate(signals):
            current_price = df.iloc[index]['close']
            current_confidence = 0  # 需要從信號中提取
            
            # 幻趥位置
            if current_position == PositionType.NEUTRAL and signal == 1:
                current_position = PositionType.LONG
                entry_price = current_price
                entry_index = index
                entry_confidence = current_confidence
            
            # 窗趥位置
            elif current_position == PositionType.NEUTRAL and signal == -1:
                current_position = PositionType.SHORT
                entry_price = current_price
                entry_index = index
                entry_confidence = current_confidence
            
            # 平什位置 (幻趥 -> 中性)
            elif current_position == PositionType.LONG and signal == -1:
                exit_price = current_price
                pnl_percent = (exit_price - entry_price) / entry_price
                pnl = pnl_percent - self.commission
                
                trade = Trade(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=df.index[entry_index],
                    exit_time=df.index[index],
                    position_type=PositionType.LONG,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    confidence=entry_confidence
                )
                self.trades.append(trade)
                
                current_position = PositionType.NEUTRAL
                self.equity_curve.append(self.equity_curve[-1] * (1 + pnl))
            
            # 平什位置 (窗趥 -> 中性)
            elif current_position == PositionType.SHORT and signal == 1:
                exit_price = current_price
                pnl_percent = (entry_price - exit_price) / entry_price  # 窗趥的敀益
                pnl = pnl_percent - self.commission
                
                trade = Trade(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=df.index[entry_index],
                    exit_time=df.index[index],
                    position_type=PositionType.SHORT,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    confidence=entry_confidence
                )
                self.trades.append(trade)
                
                current_position = PositionType.NEUTRAL
                self.equity_curve.append(self.equity_curve[-1] * (1 + pnl))
        
        # 計算汇总指標
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> BacktestResult:
        """
        計算性能指標
        
        Returns:
            BacktestResult: 回測結果
        """
        if not self.trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                trades=[],
                equity_curve=pd.Series(self.equity_curve)
            )
        
        # 基本統計
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 計算抠益率
        pnls = [t.pnl for t in self.trades]
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # 年化抠益 (250 交易日)
        trading_days = 250
        annual_return = total_return * (trading_days / (len(self.trades) + 1)) if len(self.trades) > 0 else 0
        
        # Sharpe Ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown()
        
        # 平均获利/乿損
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # 装市系数
        total_win = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = total_win / total_loss if total_loss > 0 else 0
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades,
            equity_curve=pd.Series(self.equity_curve)
        )
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        計算 Sharpe 比
        
        Args:
            risk_free_rate: 無風險利率 (預設 2%)
        
        Returns:
            float: Sharpe 比
        """
        if len(self.trades) < 2:
            return 0
        
        returns = [t.pnl for t in self.trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        sharpe = (avg_return - risk_free_rate) / std_return * np.sqrt(252)  # 年化
        return sharpe
    
    def _calculate_max_drawdown(self) -> float:
        """
        計算最大回撤
        
        Returns:
            float: 最大回撤 (%)
        """
        if len(self.equity_curve) < 2:
            return 0
        
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return abs(max_drawdown)
    
    def print_result(self, result: BacktestResult):
        """列印回測結果"""
        print("\n" + "=" * 70)
        print("回測結果")
        print("=" * 70)
        
        print(f"\n交易統計:")
        print(f"  統計次数: {result.total_trades}")
        print(f"  勝利次數: {result.winning_trades}")
        print(f"  楊摻次數: {result.losing_trades}")
        print(f"  勝率: {result.win_rate*100:.2f}%")
        
        print(f"\n計算表現:")
        print(f"  總抠益: {result.total_return*100:.2f}%")
        print(f"  年化抠益: {result.annual_return*100:.2f}%")
        print(f"  Sharpe 比: {result.sharpe_ratio:.2f}")
        print(f"  最大回撤: {result.max_drawdown*100:.2f}%")
        
        print(f"\n交易詳治:")
        print(f"  平均获利: {result.avg_win*100:.2f}%")
        print(f"  平均乿損: {result.avg_loss*100:.2f}%")
        print(f"  装市系数: {result.profit_factor:.2f}")
        
        print("\n" + "=" * 70 + "\n")
